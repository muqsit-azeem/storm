//
// Created by Muqsit Azeem on 11.08.24.
//

#include "storm-pomdp/modelchecker/POMDPExplicitDtFscModelChecker.h"
namespace storm {
namespace pomdp {
//template<typename ValueType>
//PomdpExplicitDtFscModelChecker<ValueType>::PomdpExplicitDtFScModelChecker(storm::models::sparse::Pomdp<ValueType> pomdp, ObservationPolicyPosteriorMealy policy)
//    : inputPomdp(pomdp), inputPolicy(policy) {
//    STORM_LOG_ERROR_COND(inputPomdp->isCanonic(), "Input Pomdp is not known to be canonic. This might lead to unexpected verification results.");
//    STORM_LOG_ERROR_COND(inputPomdp->hasObservationValuations(), "Explicit policy cannot be applied if POMDP was built without observation valuation!");
//    STORM_LOG_ERROR_COND(inputPomdp->hasChoiceLabeling(), "Explicit policy cannot be applied if POMDP was built without choice labeling!");
//}


template<typename ValueType>
std::shared_ptr<storm::models::sparse::Dtmc<ValueType>> PomdpExplicitDtFscModelChecker<ValueType>::buildInducedMarkovChain(
    storm::models::sparse::Pomdp<ValueType> const& pomdp, storm::pomdp::analysis::FormulaInformation const& formulaInfo) {
    std::vector<std::pair<uint64_t, uint64_t>> mcStateToPomdpStateMemoryNodeMap(pomdp.getNumberOfStates() * inputPolicy.numberOfNodes);
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> pomdpStateMemoryNodeToMCStateMap;
    std::set<uint64_t> mcStatesToExplore;
    std::set<uint64_t> exploredMCStates;

    std::unordered_map<uint64_t, std::unordered_map<uint64_t, ValueType>> transitions;
    uint64_t nextId = 0;
    uint64_t currentMCState = nextId;
    ++nextId;
    // TODO put this in its own class
    uint64_t initialPomdpState = pomdp.getInitialStates().getNextSetIndex(0);
    uint64_t initialMemoryNode = inputPolicy.initialNode;
    mcStateToPomdpStateMemoryNodeMap.at(currentMCState) = {initialPomdpState, initialMemoryNode};
    pomdpStateMemoryNodeToMCStateMap[initialPomdpState][initialMemoryNode] = currentMCState;
    mcStatesToExplore.insert(currentMCState);

    uint64_t currentPomdpState;
    uint64_t currentPolicyNode;

    std::optional<storm::storage::SparseMatrix<ValueType>> transition;

    // Collect transitions by exploring the product MC
    while (!mcStatesToExplore.empty()) {
        currentMCState = *mcStatesToExplore.begin();
        mcStatesToExplore.erase(currentMCState);
        exploredMCStates.insert(currentMCState);

        currentPomdpState = mcStateToPomdpStateMemoryNodeMap.at(currentMCState).first;
        currentPolicyNode = mcStateToPomdpStateMemoryNodeMap.at(currentMCState).second;

        if (targetObservations.count(pomdp.getObservation(currentPomdpState)) > 0) {
            // Add self-loop for goal states
            transitions[currentMCState][currentMCState] = storm::utility::one<ValueType>();
        } else {
            std::unordered_map<uint64_t, ValueType> successorStateProbabilities;
            auto successorNodeAndActionDistribution = getSuccessorNodeAndActionDistribution(currentPomdpState, currentPolicyNode);
            uint64_t successorNode = successorNodeAndActionDistribution.first;
            for (auto const& entry : successorNodeAndActionDistribution.second) {
                ValueType probability = entry.second;
                uint64_t localActionIndex = entry.first;
                for (auto const& pomdpTransition : pomdp.getTransitionMatrix().getRow(currentPomdpState, localActionIndex)) {
                    // STORM_PRINT("POMDP " << pomdpTransition << "\n")
                    if (!storm::utility::isZero(pomdpTransition.getValue())) {
                        if (successorStateProbabilities.count(pomdpTransition.getColumn()) > 0) {
                            successorStateProbabilities.at(pomdpTransition.getColumn()) += pomdpTransition.getValue() * probability;
                        } else {
                            successorStateProbabilities[pomdpTransition.getColumn()] = pomdpTransition.getValue() * probability;
                        }
                    }
                }
            }
            for (auto const& entry : successorStateProbabilities) {
                if (pomdpStateMemoryNodeToMCStateMap.count(entry.first) == 0 || pomdpStateMemoryNodeToMCStateMap.at(entry.first).count(successorNode) == 0) {
                    pomdpStateMemoryNodeToMCStateMap[entry.first][successorNode] = nextId;
                    mcStatesToExplore.insert(nextId);
                    mcStateToPomdpStateMemoryNodeMap.at(nextId) = {entry.first, successorNode};
                    ++nextId;
                }
                STORM_LOG_DEBUG("Transition: " << currentMCState << " (" << currentPomdpState << "," << currentPolicyNode << ") -- " << entry.second << " --> "
                                               << pomdpStateMemoryNodeToMCStateMap.at(entry.first).at(successorNode) << " (" << entry.first << ","
                                               << successorNode << ")");
                transitions[currentMCState][pomdpStateMemoryNodeToMCStateMap.at(entry.first).at(successorNode)] = entry.second;
            }
        }
    }

    uint64_t numberExploredStates = nextId;
    mcStateToPomdpStateMemoryNodeMap.resize(numberExploredStates);

    std::vector<ValueType> rowSums(transitions.size(), storm::utility::zero<ValueType>());
    uint64_t entryCount = 0;
    for (auto const& row : transitions) {
        entryCount += row.second.size();
        for (auto const& prob : row.second) {
            rowSums.at(row.first) += prob.second;
        }
    }

    // Normalise if we are not in the exact setting to counter numerical issues
    if (!storm::NumberTraits<ValueType>::IsExact) {
        for (auto const& row : transitions) {
            for (auto const& entry : row.second) {
                transitions[row.first][entry.first] = entry.second / rowSums.at(row.first);
            }
        }
    }

    storm::storage::SparseMatrixBuilder<ValueType> builder(transitions.size(), transitions.size(), entryCount, true, false, transitions.size());
    for (uint64_t state = 0; state < numberExploredStates; ++state) {
        for (auto const& entry : transitions[state]) {
            builder.addNextValue(state, entry.first, entry.second);
        }
    }
    auto mcTransitionMatrix = builder.build();

    storm::models::sparse::StateLabeling labeling(numberExploredStates);
    for (auto const& labelName : pomdp.getStateLabeling().getLabels()) {
        labeling.addLabel(labelName);
        // The init label is only assigned to unfolding states with the initial memory state
        if (labelName == "init") {
            for (auto const& pomdpState : pomdp.getStateLabeling().getStates(labelName)) {
                for (auto const& memNodeMcState : pomdpStateMemoryNodeToMCStateMap.at(pomdpState)) {
                    if (memNodeMcState.first == inputPolicy.initialNode) {
                        labeling.addLabelToState(labelName, memNodeMcState.second);
                    }
                }
            }
        } else {
            for (auto const& pomdpState : pomdp().getStateLabeling().getStates(labelName)) {
                if (pomdpStateMemoryNodeToMCStateMap.count(pomdpState) > 0) {
                    for (auto const& memNodeMcState : pomdpStateMemoryNodeToMCStateMap.at(pomdpState)) {
                        labeling.addLabelToState(labelName, memNodeMcState.second);
                    }
                }
            }
        }
    }

    storm::storage::sparse::ModelComponents<ValueType> components;
    components.transitionMatrix = mcTransitionMatrix;
    components.stateLabeling = labeling;

    return std::make_shared<storm::models::sparse::Dtmc<ValueType>>(std::move(components));
}


template<typename ValueType>
std::pair<uint64_t, storm::storage::Distribution<ValueType, uint64_t>>
PomdpExplicitDtFscModelChecker<ValueType>::getSuccessorNodeAndActionDistribution( storm::models::sparse::Pomdp<ValueType> const& pomdp, uint64_t currentPomdpState, uint64_t currentPolicyNode) {
    auto observationValuation = pomdp.getObservationValuations().at(pomdp.getObservation(currentPomdpState));
    auto policyForCurrentNode = inputPolicy(currentPolicyNode);
    std::unordered_map<std::string, std::string> observationInMapFormat;

    for (auto iter = observationValuation.begin(); iter != observationValuation.end(); ++iter) {
        std::string varName = iter.getName();
        std::string valString;
        if (iter.isBoolean()) {
            valString = iter.getBooleanValue() ? "TRUE" : "FALSE";
        } else if (iter.isInteger()) {
            valString = std::to_string(iter.getIntegerValue());
        } else if (iter.isRational()) {
            STORM_LOG_WARN("RATIONAL OBSERVABLES ARE NOT HANDLED YET!");
        } else if (iter.isLabelAssignment()) {
            valString = std::to_string(iter.getLabelValue());
        }
        boost::trim(varName);
        boost::trim(valString);
        observationInMapFormat[varName] = valString;
    }
    storm::storage::Distribution<typename ValueType, uint64_t> outputDistribution;
    uint64_t observationIdInPolicy;
    auto iter = std::find(inputPolicy.observations.begin(), inputPolicy.observations.end(), observationInMapFormat);
    if (iter != inputPolicy.observations.end()) {
        observationIdInPolicy = std::distance(inputPolicy.observations.begin(), iter);
        if (policyForCurrentNode.count(observationIdInPolicy) > 0) {
            uint64_t successorNode = policyForCurrentNode.at(observationIdInPolicy).first;
            storm::storage::Distribution<ValueType, uint64_t> policyDistribution =
                policyForCurrentNode.at(observationIdInPolicy).second;
            for (auto const& entry : policyDistribution) {
                typename ValueType probability = entry.second;
                std::string chosenActionName = inputPolicy.idToActionMap.at(entry.first);
                if (chosenActionName == "__target_reached") {
                    // For the target action, we default
                    outputDistribution.addProbability(0, storm::utility::one<ValueType>());
                    return {currentPolicyNode, outputDistribution};
                }
                bool actionFound = false;
                auto rowIndex = pomdp().getTransitionMatrix().getRowGroupIndices()[currentPomdpState];
                for (uint64_t i = 0; i < pomdp().getNumberOfChoices(currentPomdpState); ++i) {
                    std::string actionLabel;
                    if (pomdp().getChoiceLabeling().getLabelsOfChoice(rowIndex + i).empty()) {
                        actionLabel = "";
                    } else {
                        actionLabel = *(pomdp().getChoiceLabeling().getLabelsOfChoice(rowIndex + i).begin());
                    }
                    if (actionLabel == chosenActionName) {
                        // add the entry (with action index in local frame)
                        outputDistribution.addProbability(i, probability);
                        actionFound = true;
                        break;
                    }
                }
//                STORM_LOG_THROW(actionFound, storm::exceptions::WrongFormatException,
//                                "Action \"" << chosenActionName << "\" is not valid for POMDP state " << currentPomdpState << "!");
            }
            return {successorNode, outputDistribution};
        } else {
            // No specification for this observation in the current node
            outputDistribution.addProbability(0, storm::utility::one<ValueType>());
            return {currentPolicyNode, outputDistribution};
        }
    }
    // Observation is not mapped, play default
    outputDistribution.addProbability(0, storm::utility::one<ValueType>());
    return {currentPolicyNode, outputDistribution};
}


}  // namespace pomdp
}  // namespace storm


