#include <sstream>
#include <vector>
#include <filesystem>

#include "storm/exceptions/UnexpectedException.h"
#include "storm/models/sparse/Pomdp.h"
#include "storm/solver/SmtSolver.h"
#include "storm/storage/expressions/Expressions.h"
#include "storm/utility/Stopwatch.h"
#include "storm/utility/solver.h"

#include "storm-pomdp/analysis/WinningRegion.h"
#include "storm-pomdp/analysis/WinningRegionQueryInterface.h"


namespace storm {
namespace pomdp {

enum class MemlessSearchPathVariables { BooleanRanking, IntegerRanking, RealRanking };
MemlessSearchPathVariables pathVariableTypeFromString(std::string const& in) {
    if (in == "int") {
        return MemlessSearchPathVariables::IntegerRanking;
    } else if (in == "real") {
        return MemlessSearchPathVariables::RealRanking;
    } else {
        assert(in == "bool");
        return MemlessSearchPathVariables::BooleanRanking;
    }
}

class MemlessSearchOptions {
   public:
    void setExportSATCalls(std::string const& path) {
        exportSATcalls = path;
    }

    std::string const& getExportSATCallsPath() const {
        return exportSATcalls;
    }

    bool isExportSATSet() const {
        return exportSATcalls != "";
    }

    void setDebugLevel(uint64_t level = 1) {
        debugLevel = level;
    }

    bool computeInfoOutput() const {
        return debugLevel > 0;
    }

    bool computeDebugOutput() const {
        return debugLevel > 1;
    }

    bool computeTraceOutput() const {
        return debugLevel > 2;
    }

    void setWinningRegionFileName(std::string filename) {
        winningRegionFileName = filename;
        // return winningRegionFileName;
    }

    std::string getWinningRegionFolder() const {
        std::size_t lastDash = winningRegionFileName.find_last_of('-');
        if (lastDash != std::string::npos) {
            return winningRegionFileName.substr(0, lastDash);
        }
        return winningRegionFileName;
    }

    bool onlyDeterministicStrategies = false;
    bool forceLookahead = false;
    bool validateEveryStep = false;
    bool validateResult = false;
    MemlessSearchPathVariables pathVariableType = MemlessSearchPathVariables::RealRanking;
    uint64_t restartAfterNIterations = 250;
    uint64_t extensionCallTimeout = 0u;
    uint64_t localIterationMaximum = 600;

   private:
    std::string exportSATcalls = "";
    std::string winningRegionFileName = "";
    uint64_t debugLevel = 0;
};


struct ObservationSchedulerMoore {
        uint64_t initialNode;
        // next memory function <memory, observation> -> memory
        std::unordered_map<uint64_t, std::unordered_map<uint64_t , uint64_t>> nextMemoryTransition;
        // action selection function <memory, observation> -> action
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::vector<std::string>>> actionSelection;

        void exportMooreScheduler(ObservationSchedulerMoore schedulerMoore, const storage::sparse::StateValuations& obsValuations, std::string folderName) const {
//            STORM_PRINT("THE FINAL MEMORY FUNCTION: " << std::endl);
//            for (const auto& outerPair : schedulerMoore.nextMemoryTransition) {
//                uint64_t memory = outerPair.first;
//                const auto& transitions = outerPair.second;
//                for (const auto& innerPair : transitions) {
//                    uint64_t observation = innerPair.first;
//                    uint64_t nextMemory = innerPair.second;
//                    STORM_PRINT("Memory: " << memory
//                                           << ", Observation: " << observation
//                                           << ", Next Memory: " << nextMemory << std::endl);
//                }
//            }

            // std::string folderName = folder;
            std::string folderSchName = folderName + "/" + "schedulers";
            STORM_PRINT_AND_LOG("FOLDER SCH NAME: " << folderSchName << std::endl);
            std::string folderMemName = folderName + "/" + "memory-transitions";
            std::filesystem::create_directory(folderName);
            std::filesystem::create_directory(folderSchName);
            std::filesystem::create_directory(folderMemName);

            std::ofstream logFSC(folderName + "/" + "mem_fun.dot");
            std::ofstream logFSCTransitionsForDT(folderName + "/" + "mem_fun.csv");
            std::ofstream logActionMapping(folderName + "/" + "action_mapping.txt");
            std::ofstream OrderObservations(folderName + "/" + "ordered_observations.txt");
            bool writtenObservations = false; // check if observations are written

            if (!logFSC.is_open() || !logActionMapping.is_open() || !logFSCTransitionsForDT.is_open()) {
                std::cerr << "Failed to open scheduler files" << std::endl;
                return;
            }

            std::map<std::string, int> actionMapping;
            int actionCounter = 0;

            auto obsInfoSize = 0;
            if (!obsValuations.isEmpty(0)) { // assuming state_index 0 is valid
                auto obsInfo = obsValuations.getObsevationValuationforExplainability(0); // Assuming state_index 0
                obsInfoSize = obsInfo.size();
            }

            // Prepending the metadata to the scheduler file
            logFSCTransitionsForDT << "#PERMISSIVE" << std::endl << "BEGIN " << obsInfoSize+1 << " 1" << std::endl;

            // Writing the DOT graph header
            logFSC << "digraph MemoryTransitions {" << std::endl;

            // Adding the initial state node
            logFSC << "    \"initial\" [shape=point, width=0];" << std::endl;
            logFSC << "    \"initial\" -> \"" << schedulerMoore.initialNode << "\";" << std::endl;
            // A map to store grouped transitions
            std::map<std::pair<int, int>, std::set<std::string>> groupedTransitions;

            // Memory update
            for (const auto& [mem, nextMemFun] : schedulerMoore.nextMemoryTransition) {
                for (const auto& [obs, nextMem] : nextMemFun) {
                    std::stringstream ss;
                    std::stringstream ssDTTransitions;
                    // write source memory for DT transitions
                    ssDTTransitions << mem << ",";
                    auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
                    for (const auto& [obsName, obsVal] : obsInfo) {
                        if (ss.tellp() > 0) ss << ", ";
                        ss <<  obsName << "=" << obsVal;
                        // write observation values for DT transitions
                        ssDTTransitions << obsVal << ",";

                        // write observation names once
                        if (!writtenObservations) {
                            OrderObservations << obsName << std::endl;
                        }
                    }
                    // write destination memory for DT transitions
                    ssDTTransitions << nextMem << std::endl;
                    logFSCTransitionsForDT << ssDTTransitions.str();
                    groupedTransitions[{mem, nextMem}].insert(ss.str());
                    writtenObservations = true;
                }
            }
            logFSCTransitionsForDT.close();

            // Check if the logFSCTransitionsForDT has less than or equal to 2 lines
            std::ifstream checkFile(folderName + "/" + "mem_fun.csv");
            std::string line;
            int lineCount = 0;
            while (std::getline(checkFile, line)) {
                lineCount++;
            }
            if (lineCount <= 2) {
                // throw std::runtime_error("Error: mem_fun.csv has less than or equal to 2 lines.");
            }
            checkFile.close();

            // Writing transitions to dot file
            for (const auto& [nodes, labels] : groupedTransitions) {
                const auto& [mem, nextMem] = nodes;
                std::stringstream ss;
                //TODO: uncomment or use a different way od transition representation
//                for (const auto& label : labels) {
//                    if (ss.tellp() > 0) ss << "; ";
//                    ss << label;
//                }
                logFSC << "    \"" << mem << "\" -> \"" << nextMem << "\" [label=\"" << ss.str() << "\"];" << std::endl;
            }
            logFSC << "}" << std::endl;
            logFSC.close();
            STORM_PRINT("WRITING THE MEMORY FUNCTION FILE: " << folderName + "/" + "mem_fun.dot" << std::endl);


            // Observation based strategy
            for (const auto& [mem, ObsAction] : schedulerMoore.actionSelection) {
                // auto controllerFileName = folderName + "/" + "scheduler_" + std::to_string(mem) + ".csv";
                auto controllerFileName = folderSchName + "/" + std::to_string(mem) + ".csv";
                std::ofstream logSchedulerI(controllerFileName);
                if (!logSchedulerI.is_open()) {
                    std::cerr << "Failed to open scheduler file: " << controllerFileName << std::endl;
                    continue;
                }
                // Prepending the metadata to the scheduler file
                logSchedulerI << "#PERMISSIVE" << std::endl << "BEGIN " << obsInfoSize+1 << " 1" << std::endl;
                for (const auto& [obs, actDist] : ObsAction) {
                    // std::stringstream ssMem;
                    std::stringstream ss;
                    // todo: completely remove the memory here because we know which memory location we are in
                    if (!actDist.empty()) {
                        auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
//                        ss << mem;
//                        for (const auto& [obsName, obsVal] : obsInfo) {
//                            ss << "," << obsVal;
//                        }
//                        ss << ",";
                        // if (actDist.size() > 1) {
                            // STORM_PRINT("Multiple actions for observation: " << obs << std::endl);
                            for (const auto& act : actDist) {
                                ss << mem;
                                for (const auto& [obsName, obsVal] : obsInfo) {
                                    ss << "," << obsVal;
                                }
                                ss << ",";
                                if (actionMapping.find(act) == actionMapping.end()) {
                                    actionMapping[act] = actionCounter++;
                                }
                                int actionNumber = actionMapping[act];
                                // ss << act << ",";
                                ss << actionNumber << std::endl;
                            }
                        // }
//                        else {
//                            for (const auto& act : actDist) {
//                                if (actionMapping.find(act) == actionMapping.end()) {
//                                    actionMapping[act] = actionCounter++;
//                                }
//                                int actionNumber = actionMapping[act];
//                                // todo:delete next line
//                                ss << act << ",";
//                                ss << actionNumber;
//                            }
//                        }
                        logSchedulerI << ss.str();
                        // auto it = schedulerMoore.nextMemoryTransition.find(mem)->second;
                    }
                }
                logSchedulerI.close();

                // Check if the logSchedulerI has less than or equal to 2 lines
                checkFile.open(controllerFileName);
                lineCount = 0;
                while (std::getline(checkFile, line)) {
                    lineCount++;
                }
                if (lineCount <= 2) {
                    // throw std::runtime_error("Error: " + controllerFileName + " has less than or equal to 2 lines.");
                }
                checkFile.close();
                STORM_PRINT("WRITING THE CONTROLLER FILE: " << controllerFileName<< " for memory: " << mem << std::endl);
            }


            // memory-state transition-file
            for (const auto& [mem, ObsNextMem] : schedulerMoore.nextMemoryTransition) {
                // memory-state transition-file
                auto memoryTransitionsFileName = folderMemName + "/" + std::to_string(mem) + ".csv";
                // memory-state transition-file
                std::ofstream logMemoryTransitionsI(memoryTransitionsFileName);
                if (!logMemoryTransitionsI.is_open()) {
                    std::cerr << "Failed to open scheduler file: " << memoryTransitionsFileName << std::endl;
                    continue;
                }
                // metadata to the memory transitions file
                logMemoryTransitionsI << "#PERMISSIVE" << std::endl << "BEGIN " << obsInfoSize+1 << " 1" << std::endl;
                for (const auto& [obs, nextMem] : ObsNextMem) {
                    if (!ObsNextMem.empty()) {
                        std::stringstream ss;
                        auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
                        ss << mem;
                        for (const auto& [obsName, obsVal] : obsInfo) {
                            // write observation values for DT transitions
                            ss << "," << obsVal;
                            // STORM_PRINT("OBSINFO: " << obs << ", name=val: "<< obsName << " = " << obsVal << std::endl);
                        }
                        ss << ",";
                        ss << nextMem;
                        logMemoryTransitionsI << ss.str() << std::endl;
                    }
                }
                logMemoryTransitionsI.close();
                // Check if the logMemoryTransitionsI has less than or equal to 2 lines
                checkFile.open(memoryTransitionsFileName);
                lineCount = 0;
                while (std::getline(checkFile, line)) {
                    lineCount++;
                }
                if (lineCount <= 2) {
                    // throw std::runtime_error("Error: " + memoryTransitionsFileName + " has less than or equal to 2 lines.");
                }
                checkFile.close();
                STORM_PRINT("WRITING THE Memory FILE: " << memoryTransitionsFileName<< " for memory: " << mem << std::endl);
            }

            // Export action mappings to the file
            for (const auto& [actionName, actionNumber] : actionMapping) {
                logActionMapping << actionName << " <-> " << actionNumber << std::endl;
            }
        }
};

struct InternalObservationScheduler {
    std::vector<storm::storage::BitVector> actions;
    std::vector<uint64_t> schedulerRef;
    storm::storage::BitVector switchObservations;


    void reset(uint64_t nrObservations, uint64_t nrActions) {
        actions = std::vector<storm::storage::BitVector>(nrObservations, storm::storage::BitVector(nrActions));
        schedulerRef = std::vector<uint64_t>(nrObservations, 0);
        switchObservations.clear();
    }

    bool empty() const {
        return actions.empty();
    }

    void printForObservations(const storage::sparse::StateValuations& obsValuations, const models::sparse::ChoiceLabeling& choiceLabelling, const std::vector<uint_fast64_t>& choiceIndices,  const std::vector<std::vector<uint64_t>>& statesPerObservation, storm::storage::BitVector const& observations, storm::storage::BitVector const& observationsAfterSwitch) const {

        for (uint64_t obs = 0; obs < observations.size(); ++obs) {

            if (observations.get(obs)) {
                auto obsInfo = obsValuations.getStateInfo(obs);
                std::stringstream ss;
                ss << "Observation = " << obsInfo <<  " | Storm internal id = " << obs << " | actions = ";
                auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
                    for (auto act : actions[obs]) {
                    uint_fast64_t rowIndex = choiceIndices[stateId] + act;
                    auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
                    for (const auto& choiceLabel : choiceLabels) {
                        ss << " " << choiceLabel;
                    }
                }
                if (switchObservations.get(obs)) {
                    ss << " and switch.";
                }
                STORM_LOG_INFO(ss.str());
            }
            if (observationsAfterSwitch.get(obs)) {
                auto obsInfo = obsValuations.getStateInfo(obs);
                std::stringstream ss;
                ss << "Observation after switch = " << obsInfo <<  " | Storm internal id = " << obs << " | ";
                STORM_LOG_INFO(ss.str() << "scheduler ref: " << schedulerRef[obs]);
            }
        }
    }

    ObservationSchedulerMoore update_fsc_moore(const models::sparse::ChoiceLabeling& choiceLabelling, const std::vector<uint_fast64_t>& choiceIndices,  const std::vector<std::vector<uint64_t>>& statesPerObservation, storm::storage::BitVector const& observations, storm::storage::BitVector const& observationsAfterSwitch, std::unordered_map<uint64_t, uint64_t> winningObservationsFirstScheduler ,ObservationSchedulerMoore schedulerMoore, uint64_t schedulerId) const {
        // STORM_PRINT("SCHEDULER ID: " << schedulerId << std::endl    );
        int primeMemoryOffset = 1000000000;
        int primeSchedulerId = primeMemoryOffset + schedulerId;

        bool isSwitch = false;
        // find-out if we have to transition to the switch state
        for (uint64_t obs = 0; obs < observations.size(); ++obs) {
            if (switchObservations.get(obs)){
                isSwitch = true;
            }
        }

        for (uint64_t obs = 0; obs < observations.size(); ++obs) {
            std::vector<std::string> actionVector;
            if (observations.get(obs)) {
                auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
                for (auto act : actions[obs]) {
                    uint_fast64_t rowIndex = choiceIndices[stateId] + act;
                    auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
                    for (const auto& choiceLabel : choiceLabels) {
                        actionVector.push_back(choiceLabel);
                    }
                }
                if (!switchObservations.get(obs)){
                    schedulerMoore.nextMemoryTransition[schedulerId][obs] = schedulerId;
                    schedulerMoore.actionSelection[schedulerId][obs] = actionVector;
                }
                else {
                    // isSwitch = true;
                    schedulerMoore.nextMemoryTransition[schedulerId][obs] = primeSchedulerId;
                    //todo: check if the action selection is correct here --
                    // we want to switch to the prime scheduler and play instead of play and switch
                    schedulerMoore.actionSelection[primeSchedulerId][obs] = actionVector;
                    // schedulerMoore.actionSelection[schedulerId][obs] = actionVector;
                }
            }

            if (observationsAfterSwitch.get(obs) && isSwitch) {
                schedulerMoore.nextMemoryTransition[primeSchedulerId][obs] = schedulerRef[obs];
            }
            if (winningObservationsFirstScheduler.find(obs) != winningObservationsFirstScheduler.end() && winningObservationsFirstScheduler[obs] != schedulerId) {
                schedulerMoore.nextMemoryTransition[schedulerId][obs] = winningObservationsFirstScheduler[obs];

            }
        }
        if(isSwitch){
            // add or replicate transitions to the `primed-memory` function and action selection
            for (uint64_t obs = 0; obs < observations.size(); ++obs) {
                // add or replicate transitions to the `primed-memory` function
                if (schedulerMoore.nextMemoryTransition[schedulerId].find(obs) != schedulerMoore.nextMemoryTransition[schedulerId].end() &&
                    schedulerMoore.nextMemoryTransition[primeSchedulerId].find(obs) == schedulerMoore.nextMemoryTransition[primeSchedulerId].end()){
                    if (schedulerMoore.nextMemoryTransition[schedulerId][obs] == schedulerId) {
                        // Self-loop transition for non-offset memory location
                        schedulerMoore.nextMemoryTransition[primeSchedulerId][obs] = primeSchedulerId;
                    } else {
                        // Copy other transition
                        schedulerMoore.nextMemoryTransition[primeSchedulerId][obs] = schedulerMoore.nextMemoryTransition[schedulerId][obs];
                    }
                }
                // replicate action selections for each memory and observation
                if (schedulerMoore.actionSelection[primeSchedulerId].find(obs) == schedulerMoore.actionSelection[primeSchedulerId].end()){
                    schedulerMoore.actionSelection[primeSchedulerId][obs] = schedulerMoore.actionSelection[schedulerId][obs];
                }
            }
        }
        schedulerMoore.initialNode = schedulerId;
        return schedulerMoore;
    }
};



template<typename ValueType>
class IterativePolicySearch {
    // Implements an extension to the Chatterjee, Chmelik, Davies (AAAI-16) paper.

   public:
    class Statistics {
       public:
        Statistics() = default;
        void print() const;

        storm::utility::Stopwatch totalTimer;
        storm::utility::Stopwatch smtCheckTimer;
        storm::utility::Stopwatch initializeSolverTimer;
        storm::utility::Stopwatch evaluateExtensionSolverTime;
        storm::utility::Stopwatch encodeExtensionSolverTime;
        storm::utility::Stopwatch updateNewStrategySolverTime;
        storm::utility::Stopwatch graphSearchTime;

        storm::utility::Stopwatch winningRegionUpdatesTimer;

        void incrementOuterIterations() {
            outerIterations++;
        }

        void incrementSmtChecks() {
            satCalls++;
        }

        uint64_t getChecks() {
            return satCalls;
        }

        uint64_t getIterations() {
            return outerIterations;
        }

        uint64_t getGraphBasedwinningObservations() {
            return graphBasedAnalysisWinOb;
        }

        void incrementGraphBasedWinningObservations() {
            graphBasedAnalysisWinOb++;
        }

       private:
        uint64_t satCalls = 0;
        uint64_t outerIterations = 0;
        uint64_t graphBasedAnalysisWinOb = 0;
    };

    IterativePolicySearch(storm::models::sparse::Pomdp<ValueType> const& pomdp, storm::storage::BitVector const& targetStates,
                          storm::storage::BitVector const& surelyReachSinkStates,

                          std::shared_ptr<storm::utility::solver::SmtSolverFactory>& smtSolverFactory, MemlessSearchOptions const& options);

    bool analyzeForInitialStates(uint64_t k) {
        stats.totalTimer.start();
        STORM_LOG_TRACE("Bad states: " << surelyReachSinkStates);
        STORM_PRINT("Bad states: " << surelyReachSinkStates);
        STORM_LOG_TRACE("Target states: " << targetStates);
        STORM_PRINT("Target states: " << targetStates);
        STORM_LOG_TRACE("Questionmark states: " << (~surelyReachSinkStates & ~targetStates));
        STORM_PRINT("Questionmark tates: " << (~surelyReachSinkStates & ~targetStates));
        bool result = analyze(k, ~surelyReachSinkStates & ~targetStates, pomdp.getInitialStates());
        stats.totalTimer.stop();
        return result;
    }

    void computeWinningRegion(uint64_t k) {
        stats.totalTimer.start();
        analyze(k, ~surelyReachSinkStates & ~targetStates);
        stats.totalTimer.stop();
    }

    WinningRegion const& getLastWinningRegion() const {
        return winningRegion;
    }

    uint64_t getOffsetFromObservation(uint64_t state, uint64_t observation) const;

    void printAllValuation() const;

    bool analyze(uint64_t k, storm::storage::BitVector const& oneOfTheseStates,
                 storm::storage::BitVector const& allOfTheseStates = storm::storage::BitVector());

    Statistics const& getStatistics() const;
    void finalizeStatistics();

   private:
    storm::expressions::Expression const& getDoneActionExpression(uint64_t obs) const;

    void reset() {
        STORM_LOG_INFO("Reset solver to restart with current winning region");
        schedulerForObs.clear();
        finalSchedulers.clear();
        smtSolver->reset();
    }
    void printScheduler(std::vector<InternalObservationScheduler> const&);
    void coveredStatesToStream(std::ostream& os, storm::storage::BitVector const& remaining) const;

    bool initialize(uint64_t k);

    bool smtCheck(uint64_t iteration, std::set<storm::expressions::Expression> const& assumptions = {});

    std::unique_ptr<storm::solver::SmtSolver> smtSolver;
    storm::models::sparse::Pomdp<ValueType> const& pomdp;
    std::shared_ptr<storm::expressions::ExpressionManager> expressionManager;
    uint64_t maxK = std::numeric_limits<uint64_t>::max();

    storm::storage::BitVector surelyReachSinkStates;
    storm::storage::BitVector targetStates;
    std::vector<std::vector<uint64_t>> statesPerObservation;

    std::vector<storm::expressions::Variable> schedulerVariables;
    std::vector<storm::expressions::Expression> schedulerVariableExpressions;
    std::vector<std::vector<storm::expressions::Expression>> actionSelectionVarExpressions;  // A_{z,a}
    std::vector<std::vector<storm::expressions::Variable>> actionSelectionVars;              // A_{z,a}

    std::vector<storm::expressions::Variable> reachVars;
    std::vector<storm::expressions::Expression> reachVarExpressions;
    std::vector<std::vector<storm::expressions::Expression>> reachVarExpressionsPerObservation;

    std::vector<storm::expressions::Variable> observationUpdatedVariables;
    std::vector<storm::expressions::Expression> observationUpdatedExpressions;

    std::vector<storm::expressions::Variable> switchVars;
    std::vector<storm::expressions::Expression> switchVarExpressions;
    std::vector<storm::expressions::Variable> followVars;
    std::vector<storm::expressions::Expression> followVarExpressions;
    std::vector<storm::expressions::Variable> continuationVars;
    std::vector<storm::expressions::Expression> continuationVarExpressions;
    std::vector<std::vector<storm::expressions::Variable>> pathVars;
    std::vector<std::vector<storm::expressions::Expression>> pathVarExpressions;

    std::vector<InternalObservationScheduler> finalSchedulers;
    std::vector<uint64_t> schedulerForObs;
    WinningRegion winningRegion;

    MemlessSearchOptions options;
    Statistics stats;

    std::shared_ptr<storm::utility::solver::SmtSolverFactory>& smtSolverFactory;
    std::shared_ptr<WinningRegionQueryInterface<ValueType>> validator;

    mutable bool useFindOffset = false;
    std::string getObservationValuation(uint64_t observation);
    std::string getActionLabel(uint64_t observation, uint64_t action_offset);
};
}  // namespace pomdp
}  // namespace storm
