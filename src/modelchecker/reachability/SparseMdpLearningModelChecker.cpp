#include "src/modelchecker/reachability/SparseMdpLearningModelChecker.h"

#include "src/storage/SparseMatrix.h"
#include "src/storage/MaximalEndComponentDecomposition.h"

#include "src/logic/FragmentSpecification.h"

#include "src/utility/prism.h"

#include "src/modelchecker/results/ExplicitQuantitativeCheckResult.h"

#include "src/models/sparse/StandardRewardModel.h"

#include "src/settings/SettingsManager.h"
#include "src/settings/modules/GeneralSettings.h"

#include "src/utility/macros.h"
#include "src/exceptions/InvalidPropertyException.h"
#include "src/exceptions/NotSupportedException.h"

namespace storm {
    namespace modelchecker {
        template<typename ValueType>
        SparseMdpLearningModelChecker<ValueType>::SparseMdpLearningModelChecker(storm::prism::Program const& program, boost::optional<std::map<storm::expressions::Variable, storm::expressions::Expression>> const& constantDefinitions) : program(storm::utility::prism::preprocessProgram<ValueType>(program, constantDefinitions)), variableInformation(this->program), //randomGenerator(std::chrono::system_clock::now().time_since_epoch().count()),
        comparator(1e-9) {
            // Intentionally left empty.
        }
        
        template<typename ValueType>
        bool SparseMdpLearningModelChecker<ValueType>::canHandle(CheckTask<storm::logic::Formula> const& checkTask) const {
            storm::logic::Formula const& formula = checkTask.getFormula();
            storm::logic::FragmentSpecification fragment = storm::logic::propositional().setProbabilityOperatorsAllowed(true).setReachabilityProbabilityFormulasAllowed(true).setNestedOperatorsAllowed(false);
            return formula.isInFragment(fragment) && checkTask.isOnlyInitialStatesRelevantSet();
        }
        
        template<typename ValueType>
        std::unique_ptr<CheckResult> SparseMdpLearningModelChecker<ValueType>::computeReachabilityProbabilities(CheckTask<storm::logic::EventuallyFormula> const& checkTask) {
            storm::logic::EventuallyFormula const& eventuallyFormula = checkTask.getFormula();
            storm::logic::Formula const& subformula = eventuallyFormula.getSubformula();
            STORM_LOG_THROW(program.isDeterministicModel() || checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException, "For nondeterministic systems, an optimization direction (min/max) must be given in the property.");
            STORM_LOG_THROW(subformula.isAtomicExpressionFormula() || subformula.isAtomicLabelFormula(), storm::exceptions::NotSupportedException, "Learning engine can only deal with formulas of the form 'F \"label\"' or 'F expression'.");
            
            StateGeneration stateGeneration(storm::generator::PrismNextStateGenerator<ValueType, StateType>(program, variableInformation, false), getTargetStateExpression(subformula));
            
            ExplorationInformation explorationInformation(variableInformation.getTotalBitOffset(true));
            explorationInformation.optimizationDirection = checkTask.isOptimizationDirectionSet() ? checkTask.getOptimizationDirection() : storm::OptimizationDirection::Maximize;
            
            // The first row group starts at action 0.
            explorationInformation.newRowGroup(0);
            
            // Create a callback for the next-state generator to enable it to request the index of states.
            std::function<StateType (storm::generator::CompressedState const&)> stateToIdCallback = createStateToIdCallback(explorationInformation);
            
            // Compute and return result.
            std::tuple<StateType, ValueType, ValueType> boundsForInitialState = performLearningProcedure(stateGeneration, explorationInformation);
            return std::make_unique<ExplicitQuantitativeCheckResult<ValueType>>(std::get<0>(boundsForInitialState), std::get<1>(boundsForInitialState));
        }
        
        template<typename ValueType>
        storm::expressions::Expression SparseMdpLearningModelChecker<ValueType>::getTargetStateExpression(storm::logic::Formula const& subformula) const {
            storm::expressions::Expression result;
            if (subformula.isAtomicExpressionFormula()) {
                result = subformula.asAtomicExpressionFormula().getExpression();
            } else {
                result = program.getLabelExpression(subformula.asAtomicLabelFormula().getLabel());
            }
            return result;
        }
        
        template<typename ValueType>
        std::function<typename SparseMdpLearningModelChecker<ValueType>::StateType (storm::generator::CompressedState const&)> SparseMdpLearningModelChecker<ValueType>::createStateToIdCallback(ExplorationInformation& explorationInformation) const {
            return [&explorationInformation] (storm::generator::CompressedState const& state) -> StateType {
                StateType newIndex = explorationInformation.stateStorage.numberOfStates;
                
                // Check, if the state was already registered.
                std::pair<uint32_t, std::size_t> actualIndexBucketPair = explorationInformation.stateStorage.stateToId.findOrAddAndGetBucket(state, newIndex);
                
                if (actualIndexBucketPair.first == newIndex) {
                    explorationInformation.addUnexploredState(state);
                }
                
                return actualIndexBucketPair.first;
            };
        }
        
        template<typename ValueType>
        std::tuple<typename SparseMdpLearningModelChecker<ValueType>::StateType, ValueType, ValueType> SparseMdpLearningModelChecker<ValueType>::performLearningProcedure(StateGeneration& stateGeneration, ExplorationInformation& explorationInformation) const {
            
            // Generate the initial state so we know where to start the simulation.
            explorationInformation.setInitialStates(stateGeneration.getInitialStates());
            STORM_LOG_THROW(explorationInformation.getNumberOfInitialStates() == 1, storm::exceptions::NotSupportedException, "Currently only models with one initial state are supported by the learning engine.");
            StateType initialStateIndex = explorationInformation.getFirstInitialState();
            
            // Create a structure that holds the bounds for the states and actions.
            BoundValues bounds;

            // Create a stack that is used to track the path we sampled.
            StateActionStack stack;
            
            // Now perform the actual sampling.
            Statistics stats;
            bool convergenceCriterionMet = false;
            while (!convergenceCriterionMet) {
                bool result = samplePathFromState(stateGeneration, explorationInformation, stack, bounds, stats);
                
                // If a terminal state was found, we update the probabilities along the path contained in the stack.
                if (result) {
                    // Update the bounds along the path to the terminal state.
                    STORM_LOG_TRACE("Found terminal state, updating probabilities along path.");
                    updateProbabilityBoundsAlongSampledPath(stack, explorationInformation, bounds);
                } else {
                    // If not terminal state was found, the search aborted, possibly because of an EC-detection. In this
                    // case, we cannot update the probabilities.
                    STORM_LOG_TRACE("Did not find terminal state.");
                }
                
                STORM_LOG_DEBUG("Discovered states: " << explorationInformation.getNumberOfDiscoveredStates() << " (" << stats.numberOfExploredStates << "explored, " << explorationInformation.getNumberOfUnexploredStates() << " unexplored).");
                STORM_LOG_DEBUG("Value of initial state is in [" << bounds.getLowerBoundForState(initialStateIndex, explorationInformation) << ", " << bounds.getUpperBoundForState(initialStateIndex, explorationInformation) << "].");
                ValueType difference = bounds.getDifferenceOfStateBounds(initialStateIndex, explorationInformation);
                STORM_LOG_DEBUG("Difference after iteration " << stats.iterations << " is " << difference << ".");
                convergenceCriterionMet = difference < 1e-6;
                
                ++stats.iterations;
            }
            
            if (storm::settings::generalSettings().isShowStatisticsSet()) {
                std::cout << std::endl << "Learning summary -------------------------" << std::endl;
                std::cout << "Discovered states: " << explorationInformation.getNumberOfDiscoveredStates() << " (" << stats.numberOfExploredStates << "explored, " << explorationInformation.getNumberOfUnexploredStates() << " unexplored, " << stats.numberOfTargetStates << " target states)" << std::endl;
                std::cout << "Sampling iterations: " << stats.iterations << std::endl;
                std::cout << "Maximal path length: " << stats.maxPathLength << std::endl;
            }
            
            return std::make_tuple(initialStateIndex, bounds.getLowerBoundForState(initialStateIndex, explorationInformation), bounds.getUpperBoundForState(initialStateIndex, explorationInformation));
        }
        
        template<typename ValueType>
        bool SparseMdpLearningModelChecker<ValueType>::samplePathFromState(StateGeneration& stateGeneration, ExplorationInformation& explorationInformation, StateActionStack& stack, BoundValues& bounds, Statistics& stats) const {
            
            // Start the search from the initial state.
            stack.push_back(std::make_pair(explorationInformation.getFirstInitialState(), 0));
            
            bool foundTerminalState = false;
            while (!foundTerminalState) {
                StateType const& currentStateId = stack.back().first;
                STORM_LOG_TRACE("State on top of stack is: " << currentStateId << ".");
                
                // If the state is not yet explored, we need to retrieve its behaviors.
                auto unexploredIt = explorationInformation.unexploredStates.find(currentStateId);
                if (unexploredIt != explorationInformation.unexploredStates.end()) {
                    STORM_LOG_TRACE("State was not yet explored.");
                    
                    // Explore the previously unexplored state.
                    storm::generator::CompressedState const& compressedState = unexploredIt->second;
                    foundTerminalState = exploreState(stateGeneration, currentStateId, compressedState, explorationInformation, bounds, stats);
                    explorationInformation.unexploredStates.erase(unexploredIt);
                } else {
                    // If the state was already explored, we check whether it is a terminal state or not.
                    if (explorationInformation.isTerminal(currentStateId)) {
                        STORM_LOG_TRACE("Found already explored terminal state: " << currentStateId << ".");
                        foundTerminalState = true;
                    }
                }
                
                // If the state was not a terminal state, we continue the path search and sample the next state.
                if (!foundTerminalState) {
                    // At this point, we can be sure that the state was expanded and that we can sample according to the
                    // probabilities in the matrix.
                    uint32_t chosenAction = sampleFromMaxActions(currentStateId, explorationInformation, bounds);
                    stack.back().second = chosenAction;
                    STORM_LOG_TRACE("Sampled action " << chosenAction << " in state " << currentStateId << ".");
                    
                    StateType successor = sampleSuccessorFromAction(chosenAction, explorationInformation);
                    STORM_LOG_TRACE("Sampled successor " << successor << " according to action " << chosenAction << " of state " << currentStateId << ".");
                    
                    // Put the successor state and a dummy action on top of the stack.
                    stack.emplace_back(successor, 0);
                    stats.maxPathLength = std::max(stats.maxPathLength, stack.size());
                    
                    // If the current path length exceeds the threshold and the model is a nondeterministic one, we
                    // perform an EC detection.
                    if (stack.size() > stats.pathLengthUntilEndComponentDetection && !program.isDeterministicModel()) {
                        detectEndComponents(stack, explorationInformation, bounds);
                        
                        // Abort the current search.
                        STORM_LOG_TRACE("Aborting the search after EC detection.");
                        stack.clear();
                        break;
                    }
                }
            }
            
            return foundTerminalState;
        }
        
        template<typename ValueType>
        bool SparseMdpLearningModelChecker<ValueType>::exploreState(StateGeneration& stateGeneration, StateType const& currentStateId, storm::generator::CompressedState const& currentState, ExplorationInformation& explorationInformation, BoundValues& bounds, Statistics& stats) const {
            
            bool isTerminalState = false;
            bool isTargetState = false;
            
            // Before generating the behavior of the state, we need to determine whether it's a target state that
            // does not need to be expanded.
            stateGeneration.generator.load(currentState);
            if (stateGeneration.isTargetState()) {
                ++stats.numberOfTargetStates;
                isTargetState = true;
                isTerminalState = true;
            } else {
                STORM_LOG_TRACE("Exploring state.");
                
                // If it needs to be expanded, we use the generator to retrieve the behavior of the new state.
                storm::generator::StateBehavior<ValueType, StateType> behavior = stateGeneration.expand();
                STORM_LOG_TRACE("State has " << behavior.getNumberOfChoices() << " choices.");
                
                // Clumsily check whether we have found a state that forms a trivial BMEC.
                if (behavior.getNumberOfChoices() == 0) {
                    isTerminalState = true;
                } else if (behavior.getNumberOfChoices() == 1) {
                    auto const& onlyChoice = *behavior.begin();
                    if (onlyChoice.size() == 1) {
                        auto const& onlyEntry = *onlyChoice.begin();
                        if (onlyEntry.first == currentStateId) {
                            isTerminalState = true;
                        }
                    }
                }
                
                // If the state was neither a trivial (non-accepting) terminal state nor a target state, we
                // need to store its behavior.
                if (!isTerminalState) {
                    // Next, we insert the behavior into our matrix structure.
                    StateType startRow = explorationInformation.matrix.size();
                    explorationInformation.addRowsToMatrix(behavior.getNumberOfChoices());
                    
                    // Terminate the row group.
                    explorationInformation.rowGroupIndices.push_back(explorationInformation.matrix.size());
                    
                    ActionType currentAction = 0;
                    for (auto const& choice : behavior) {
                        for (auto const& entry : choice) {
                            std::cout << "adding " << currentStateId << " -> " << entry.first << " with prob " << entry.second << std::endl;
                            explorationInformation.matrix[startRow + currentAction].emplace_back(entry.first, entry.second);
                        }
                        
                        bounds.initializeActionBoundsForNextAction(computeBoundsOfAction(startRow + currentAction, explorationInformation, bounds));

                        STORM_LOG_TRACE("Initializing bounds of action " << (startRow + currentAction) << " to " << bounds.getLowerBoundForAction(startRow + currentAction) << " and " << bounds.getUpperBoundForAction(startRow + currentAction) << ".");
                        
                        ++currentAction;
                    }
                    
                    bounds.initializeStateBoundsForNextState(computeBoundsOfState(currentStateId, explorationInformation, bounds));
                    STORM_LOG_TRACE("Initializing bounds of state " << currentStateId << " to " << bounds.getLowerBoundForState(currentStateId) << " and " << bounds.getUpperBoundForState(currentStateId) << ".");
                }
            }
            
            if (isTerminalState) {
                STORM_LOG_TRACE("State does not need to be explored, because it is " << (isTargetState ? "a target state" : "a rejecting terminal state") << ".");
                explorationInformation.addTerminalState(currentStateId);
                
                if (isTargetState) {
                    bounds.initializeStateBoundsForNextState(std::make_pair(storm::utility::one<ValueType>(), storm::utility::one<ValueType>()));
                    bounds.initializeStateBoundsForNextAction(std::make_pair(storm::utility::one<ValueType>(), storm::utility::one<ValueType>()));
                } else {
                    bounds.initializeStateBoundsForNextState(std::make_pair(storm::utility::zero<ValueType>(), storm::utility::zero<ValueType>()));
                    bounds.initializeStateBoundsForNextAction(std::make_pair(storm::utility::zero<ValueType>(), storm::utility::zero<ValueType>()));
                }
                
                // Increase the size of the matrix, but leave the row empty.
                explorationInformation.addRowsToMatrix(1);
                
                // Terminate the row group.
                explorationInformation.newRowGroup();
            }
            
            // Finally, map the unexplored state to the row group.
            explorationInformation.assignStateToNextRowGroup(currentStateId);
            STORM_LOG_TRACE("Assigning row group " << explorationInformation.getRowGroup(currentStateId) << " to state " << currentStateId << ".");
            
            return isTerminalState;
        }
        
        template<typename ValueType>
        uint32_t SparseMdpLearningModelChecker<ValueType>::sampleMaxAction(StateType const& currentStateId, ExplorationInformation const& explorationInformation, BoundValues& bounds) const {
            StateType rowGroup = explorationInformation.getRowGroup(currentStateId);
            
            // First, determine all maximizing actions.
            std::vector<ActionType> allMaxActions;
            
            // Determine the values of all available actions.
            std::vector<std::pair<ActionType, ValueType>> actionValues;
            auto choicesInEcIt = explorationInformation.stateToLeavingChoicesOfEndComponent.find(currentStateId);
            if (choicesInEcIt != explorationInformation.stateToLeavingChoicesOfEndComponent.end()) {
                STORM_LOG_TRACE("Sampling from actions leaving the previously detected EC.");
                for (auto const& row : *choicesInEcIt->second) {
                    actionValues.push_back(std::make_pair(row, computeUpperBoundOfAction(row, explorationInformation, bounds)));
                }
            } else {
                STORM_LOG_TRACE("Sampling from actions leaving the state.");
                
                for (uint32_t row = explorationInformation.getStartRowOfGroup(rowGroup); row < explorationInformation.getStartRowOfGroup(rowGroup + 1); ++row) {
                    actionValues.push_back(std::make_pair(row, computeUpperBoundOfAction(row, explorationInformation, bounds)));
                }
            }
            
            std::sort(actionValues.begin(), actionValues.end(), [] (std::pair<ActionType, ValueType> const& a, std::pair<ActionType, ValueType> const& b) { return b.second > a.second; } );
            auto end = std::equal_range(actionValues.begin(), actionValues.end(), [this] (std::pair<ActionType, ValueType> const& a, std::pair<ActionType, ValueType> const& b) { return comparator.isEqual(a.second, b.second); } );
            
            // Now sample from all maximizing actions.
            std::uniform_int_distribution<uint32_t> distribution(0, std::distance(actionValues.begin(), end));
            return actionValues[distribution(randomGenerator)].first;
        }
        
        template<typename ValueType>
        typename SparseMdpLearningModelChecker<ValueType>::StateType SparseMdpLearningModelChecker<ValueType>::sampleSuccessorFromAction(ActionType const& chosenAction, ExplorationInformation const& explorationInformation) const {
            // TODO: precompute this?
            std::vector<ValueType> probabilities(explorationInformation.getRowOfMatrix(chosenAction).size());
            std::transform(explorationInformation.getRowOfMatrix(chosenAction).begin(), explorationInformation.getRowOfMatrix(chosenAction).end(), probabilities.begin(), [] (storm::storage::MatrixEntry<StateType, ValueType> const& entry) { return entry.getValue(); } );
            
            // Now sample according to the probabilities.
            std::discrete_distribution<StateType> distribution(probabilities.begin(), probabilities.end());
            StateType offset = distribution(randomGenerator);
            return explorationInformation.getRowOfMatrix(chosenAction)[offset].getColumn();
        }
        
        template<typename ValueType>
        void SparseMdpLearningModelChecker<ValueType>::detectEndComponents(StateActionStack const& stack, ExplorationInformation& explorationInformation, BoundValues& bounds) const {
            STORM_LOG_TRACE("Starting EC detection.");
            
            // Outline:
            // 1. construct a sparse transition matrix of the relevant part of the state space.
            // 2. use this matrix to compute an MEC decomposition.
            // 3. if non-empty analyze the decomposition for accepting/rejecting MECs.
            
            // Start with 1.
            storm::storage::SparseMatrixBuilder<ValueType> builder(0, 0, 0, false, true, 0);
            
            // Determine the set of states that was expanded.
            std::vector<StateType> relevantStates;
            for (StateType state = 0; state < explorationInformation.stateStorage.numberOfStates; ++state) {
                if (!explorationInformation.isUnexplored(state)) {
                    relevantStates.push_back(state);
                }
            }

            // Sort according to the actual row groups so we can insert the elements in order later.
            std::sort(relevantStates.begin(), relevantStates.end(), [&explorationInformation] (StateType const& a, StateType const& b) { return explorationInformation.getRowGroup(a) < explorationInformation.getRowGroup(b); });
            StateType unexploredState = relevantStates.size();
            
            // Create a mapping for faster look-up during the translation of flexible matrix to the real sparse matrix.
            std::unordered_map<StateType, StateType> relevantStateToNewRowGroupMapping;
            for (StateType index = 0; index < relevantStates.size(); ++index) {
                std::cout << "relevant: " << relevantStates[index] << std::endl;
                relevantStateToNewRowGroupMapping.emplace(relevantStates[index], index);
            }
            
            // Do the actual translation.
            StateType currentRow = 0;
            for (auto const& state : relevantStates) {
                builder.newRowGroup(currentRow);
                StateType rowGroup = explorationInformation.getRowGroup(state);
                for (auto row = explorationInformation.getStartRowOfGroup(rowGroup); row < explorationInformation.getStartRowOfGroup(rowGroup + 1); ++row) {
                    ValueType unexpandedProbability = storm::utility::zero<ValueType>();
                    for (auto const& entry : explorationInformation.getRowOfMatrix(row)) {
                        auto it = relevantStateToNewRowGroupMapping.find(entry.getColumn());
                        if (it != relevantStateToNewRowGroupMapping.end()) {
                            // If the entry is a relevant state, we copy it over (and compensate for the offset change).
                            builder.addNextValue(currentRow, it->second, entry.getValue());
                        } else {
                            // If the entry is an unexpanded state, we gather the probability to later redirect it to an unexpanded sink.
                            unexpandedProbability += entry.getValue();
                        }
                    }
                    if (unexpandedProbability != storm::utility::zero<ValueType>()) {
                        builder.addNextValue(currentRow, unexploredState, unexpandedProbability);
                    }
                    ++currentRow;
                }
            }
            // Then, make the unexpanded state absorbing.
            builder.newRowGroup(currentRow);
            builder.addNextValue(currentRow, unexploredState, storm::utility::one<ValueType>());
            STORM_LOG_TRACE("Successfully built matrix for MEC decomposition.");
            
            // Go on to step 2.
            storm::storage::SparseMatrix<ValueType> relevantStatesMatrix = builder.build();
            storm::storage::MaximalEndComponentDecomposition<ValueType> mecDecomposition(relevantStatesMatrix, relevantStatesMatrix.transpose(true));
            STORM_LOG_TRACE("Successfully computed MEC decomposition. Found " << (mecDecomposition.size() > 1 ? (mecDecomposition.size() - 1) : 0) << " MEC(s).");
            
            // 3. Analyze the MEC decomposition.
            for (auto const& mec : mecDecomposition) {
                // Ignore the (expected) MEC of the unexplored state.
                if (mec.containsState(unexploredState)) {
                    continue;
                }
                
                bool containsTargetState = false;
                
                // Now we record all choices leaving the EC.
                ActionSetPointer leavingChoices = std::make_shared<ActionSet>();
                for (auto const& stateAndChoices : mec) {
                    // Compute the state of the original model that corresponds to the current state.
                    std::cout << "local state: " << stateAndChoices.first << std::endl;
                    StateType originalState = relevantStates[stateAndChoices.first];
                    std::cout << "original state: " << originalState << std::endl;
                    uint32_t originalRowGroup = explorationInformation.getRowGroup(originalState);
                    std::cout << "original row group: " << originalRowGroup << std::endl;
                    
                    // TODO: This checks for a target state is a bit hackish and only works for max probabilities.
                    if (!containsTargetState && comparator.isOne(bounds.getLowerBoundForRowGroup(originalRowGroup, explorationInformation))) {
                        containsTargetState = true;
                    }
                    
                    auto includedChoicesIt = stateAndChoices.second.begin();
                    auto includedChoicesIte = stateAndChoices.second.end();
                    
                    for (auto action = explorationInformation.getStartRowOfGroup(originalRowGroup); action < explorationInformation.getStartRowOfGroup(originalRowGroup + 1); ++action) {
                        if (includedChoicesIt != includedChoicesIte) {
                            STORM_LOG_TRACE("Next (local) choice contained in MEC is " << (*includedChoicesIt - relevantStatesMatrix.getRowGroupIndices()[stateAndChoices.first]));
                            STORM_LOG_TRACE("Current (local) choice iterated is " << (action - explorationInformation.getStartRowOfGroup(originalRowGroup)));
                            if (action - explorationInformation.getStartRowOfGroup(originalRowGroup) != *includedChoicesIt - relevantStatesMatrix.getRowGroupIndices()[stateAndChoices.first]) {
                                STORM_LOG_TRACE("Choice leaves the EC.");
                                leavingChoices->insert(action);
                            } else {
                                STORM_LOG_TRACE("Choice stays in the EC.");
                                ++includedChoicesIt;
                            }
                        } else {
                            STORM_LOG_TRACE("Choice leaves the EC, because there is no more choice staying in the EC.");
                            leavingChoices->insert(action);
                        }
                    }
                    
                    explorationInformation.stateToLeavingChoicesOfEndComponent[originalState] = leavingChoices;
                }
                
                // If one of the states of the EC is a target state, all states in the EC have probability 1.
                if (containsTargetState) {
                    STORM_LOG_TRACE("MEC contains a target state.");
                    for (auto const& stateAndChoices : mec) {
                        // Compute the state of the original model that corresponds to the current state.
                        StateType originalState = relevantStates[stateAndChoices.first];
                        
                        STORM_LOG_TRACE("Setting lower bound of state in row group " << explorationInformation.getRowGroup(originalState) << " to 1.");
                        bounds.setLowerBoundForState(originalState, explorationInformation, storm::utility::one<ValueType>());
                        explorationInformation.addTerminalState(originalState);
                    }
                } else if (leavingChoices->empty()) {
                    STORM_LOG_TRACE("MEC's leaving choices are empty.");
                    // If there is no choice leaving the EC, but it contains no target state, all states have probability 0.
                    for (auto const& stateAndChoices : mec) {
                        // Compute the state of the original model that corresponds to the current state.
                        StateType originalState = relevantStates[stateAndChoices.first];
                        
                        STORM_LOG_TRACE("Setting upper bound of state in row group " << explorationInformation.getRowGroup(originalState) << " to 0.");
                        bounds.setUpperBoundForState(originalState, explorationInformation, storm::utility::zero<ValueType>());
                        explorationInformation.addTerminalState(originalState);
                    }
                }
            }
        }
        
        template<typename ValueType>
        ValueType SparseMdpLearningModelChecker<ValueType>::computeLowerBoundOfAction(ActionType const& action, ExplorationInformation const& explorationInformation, BoundValues const& bounds) const {
            ValueType result = storm::utility::zero<ValueType>();
            for (auto const& element : explorationInformation.getRowOfMatrix(action)) {
                result += element.getValue() * bounds.getLowerBoundForState(element.getColumn(), explorationInformation);
            }
            return result;
        }
        
        template<typename ValueType>
        ValueType SparseMdpLearningModelChecker<ValueType>::computeUpperBoundOfAction(ActionType const& action, ExplorationInformation const& explorationInformation, BoundValues const& bounds) const {
            ValueType result = storm::utility::zero<ValueType>();
            for (auto const& element : explorationInformation.getRowOfMatrix(action)) {
                result += element.getValue() * bounds.getUpperBoundForState(element.getColumn(), explorationInformation);
            }
            return result;
        }
        
        template<typename ValueType>
        ValueType SparseMdpLearningModelChecker<ValueType>::computeLowerBoundOfState(StateType const& state, ExplorationInformation const& explorationInformation, BoundValues const& bounds) const {
            StateType group = explorationInformation.getRowGroup(state);
            ValueType result = std::make_pair(storm::utility::zero<ValueType>(), storm::utility::zero<ValueType>());
            for (ActionType action = explorationInformation.getStartRowOfGroup(group); action < explorationInformation.getStartRowOfGroup(group + 1); ++action) {
                ValueType actionValue = computeLowerBoundOfAction(action, explorationInformation, bounds);
                result = std::max(actionValue, result);
            }
            return result;
        }
    
        template<typename ValueType>
        ValueType SparseMdpLearningModelChecker<ValueType>::computeUpperBoundOfState(StateType const& state, ExplorationInformation const& explorationInformation, BoundValues const& bounds) const {
            StateType group = explorationInformation.getRowGroup(state);
            ValueType result = std::make_pair(storm::utility::zero<ValueType>(), storm::utility::zero<ValueType>());
            for (ActionType action = explorationInformation.getStartRowOfGroup(group); action < explorationInformation.getStartRowOfGroup(group + 1); ++action) {
                ValueType actionValue = computeUpperBoundOfAction(action, explorationInformation, bounds);
                result = std::max(actionValue, result);
            }
            return result;
        }
        
        template<typename ValueType>
        std::pair<ValueType, ValueType> SparseMdpLearningModelChecker<ValueType>::computeBoundsOfAction(ActionType const& action, ExplorationInformation const& explorationInformation, BoundValues const& bounds) const {
            // TODO: take into account self-loops?
            std::pair<ValueType, ValueType> result = std::make_pair(storm::utility::zero<ValueType>(), storm::utility::zero<ValueType>());
            for (auto const& element : explorationInformation.getRowOfMatrix(action)) {
                result.first += element.getValue() * bounds.getLowerBoundForState(element.getColumn(), explorationInformation);
                result.second += element.getValue() * bounds.getUpperBoundForState(element.getColumn(), explorationInformation);
            }
            return result;
        }

        template<typename ValueType>
        std::pair<ValueType, ValueType> SparseMdpLearningModelChecker<ValueType>::computeBoundsOfState(StateType const& currentStateId, ExplorationInformation const& explorationInformation, BoundValues const& bounds) const {
            StateType group = explorationInformation.getRowGroup(currentStateId);
            std::pair<ValueType, ValueType> result = std::make_pair(storm::utility::zero<ValueType>(), storm::utility::zero<ValueType>());
            for (ActionType action = explorationInformation.getStartRowOfGroup(group); action < explorationInformation.getStartRowOfGroup(group + 1); ++action) {
                std::pair<ValueType, ValueType> actionValues = computeBoundsOfAction(action, explorationInformation, bounds);
                result.first = std::max(actionValues.first, result.first);
                result.second = std::max(actionValues.second, result.second);
            }
            return result;
        }
        
        template<typename ValueType>
        void SparseMdpLearningModelChecker<ValueType>::updateProbabilityBoundsAlongSampledPath(StateActionStack& stack, ExplorationInformation const& explorationInformation, BoundValues& bounds) const {
            std::cout << "stack is:" << std::endl;
            for (auto const& el : stack) {
                std::cout << el.first << "-[" << el.second << "]-> ";
            }
            std::cout << std::endl;
            
            stack.pop_back();
            while (!stack.empty()) {
                updateProbabilityOfAction(stack.back().first, stack.back().second, explorationInformation, bounds);
                stack.pop_back();
            }
        }

        template<typename ValueType>
        void SparseMdpLearningModelChecker<ValueType>::updateProbabilityOfAction(StateType const& state, ActionType const& action, ExplorationInformation const& explorationInformation, BoundValues& bounds) const {
            // Compute the new lower/upper values of the action.
            std::pair<ValueType, ValueType> newBoundsForAction = computeBoundsOfAction(action, explorationInformation, bounds);
            
            // And set them as the current value.
            bounds.setBoundForAction(action, newBoundsForAction);
            
            // Check if we need to update the values for the states.
            bounds.setNewLowerBoundOfStateIfGreaterThanOld(state, explorationInformation, newBoundsForAction.first);
            
            StateType rowGroup = explorationInformation.getRowGroup(state);
            if (newBoundsForAction < bounds.getUpperBoundOfRowGroup(rowGroup)) {
                
            }
            
            ValueType upperBound = computeUpperBoundOverAllOtherActions(state, action, explorationInformation, bounds);
            
            uint32_t sourceRowGroup = stateToRowGroupMapping[sourceStateId];
            if (newUpperValue < upperBoundsPerState[sourceRowGroup]) {
                if (rowGroupIndices[sourceRowGroup + 1] - rowGroupIndices[sourceRowGroup] > 1) {
                    ValueType max = storm::utility::zero<ValueType>();

                    for (uint32_t currentAction = rowGroupIndices[sourceRowGroup]; currentAction < rowGroupIndices[sourceRowGroup + 1]; ++currentAction) {
                        std::cout << "cur: " << currentAction << std::endl;
                        if (currentAction == action) {
                            continue;
                        }
                        
                        ValueType currentValue = storm::utility::zero<ValueType>();
                        for (auto const& element : transitionMatrix[currentAction]) {
                            currentValue += element.getValue() * (stateToRowGroupMapping[element.getColumn()] == unexploredMarker ? storm::utility::one<ValueType>() : upperBoundsPerState[stateToRowGroupMapping[element.getColumn()]]);
                        }
                        max = std::max(max, currentValue);
                        std::cout << "max is " << max << std::endl;
                    }
                    
                    newUpperValue = std::max(newUpperValue, max);
                }
                
                if (newUpperValue < upperBoundsPerState[sourceRowGroup]) {
                    STORM_LOG_TRACE("Got new upper bound for state " << sourceStateId << ": " << newUpperValue << " (was " << upperBoundsPerState[sourceRowGroup] << ").");
                    std::cout << "writing at index " << sourceRowGroup << std::endl;
                    upperBoundsPerState[sourceRowGroup] = newUpperValue;
                }
            }
        }
        
        template class SparseMdpLearningModelChecker<double>;
    }
}