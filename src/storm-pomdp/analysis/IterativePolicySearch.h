#include <sstream>
#include <vector>
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
    uint64_t debugLevel = 0;
};


struct ObservationSchedulerMoore {
        uint64_t initialNode;
        // next memory function <memory, observation> -> memory
        std::unordered_map<uint64_t, std::unordered_map<uint64_t , uint64_t>> nextMemoryTransition;
        // action selection function <memory, observation> -> action
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::vector<std::string>>> actionSelection;

        void exportMooreScheduler(ObservationSchedulerMoore schedulerMoore, const storage::sparse::StateValuations& obsValuations, uint64_t hash) const {
            std::string folderName = std::to_string(hash);
            std::string folderSchName = std::to_string(hash) + "/" + "schedulers";
            std::filesystem::create_directory(folderName);
            std::filesystem::create_directory(folderSchName);

            std::ofstream logFSC(folderName + "/" + "mem_fun.fsc");
            std::ofstream logActionMapping(folderName + "/" + "action_mapping.txt");

            if (!logFSC.is_open() || !logActionMapping.is_open()) {
                std::cerr << "Failed to open scheduler files" << std::endl;
                return;
            }

            std::map<std::string, int> actionMapping;
            int actionCounter = 0;

            // Memory update function
            for (const auto& [mem, nextMemFun] : schedulerMoore.nextMemoryTransition) {
                for (const auto& [obs, nextMem] : nextMemFun) {
                    std::stringstream ss;
                    ss << mem;
                    auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
                    for (const auto& [obsName, obsVal] : obsInfo) {
                        ss << "," << obsVal;
                    }
                    ss << " -> " << nextMem;
                    logFSC << ss.str() << std::endl;
                }
            }

            auto obsInfoSize = 0;
            if (!obsValuations.isEmpty(0)) { // Assuming state_index 0 is valid; adjust as needed
                auto obsInfo = obsValuations.getObsevationValuationforExplainability(0); // Assuming state_index 0
                obsInfoSize = obsInfo.size();
            }


            // Observation based strategy
            for (const auto& [mem, ObsAction] : schedulerMoore.actionSelection) {
                // auto controllerFileName = folderName + "/" + "scheduler_" + std::to_string(mem) + ".csv";
                auto controllerFileName = folderSchName + "/" + std::to_string(mem) + ".csv";
                std::ofstream logSchedulerI(controllerFileName);
                if (!logSchedulerI.is_open()) {
                    std::cerr << "Failed to open scheduler file: " << controllerFileName << std::endl;
                    continue;
                }

                int ObsActPairCounter = 0;
                for (const auto& [obs, actDist] : ObsAction) {
                    std::stringstream ss;
                    // todo: completely remove the memory here because we know which memory location we are in
                    if (!actDist.empty()) {
                        auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
                        ss << mem;
                        for (const auto& [obsName, obsVal] : obsInfo) {
                            ss << "," << obsVal;
                        }
                        ss << ",";
                        for (const auto& act : actDist) {
                            if (actionMapping.find(act) == actionMapping.end()) {
                                actionMapping[act] = actionCounter++;
                            }
                            int actionNumber = actionMapping[act];
                            ss << actionNumber;
                            ObsActPairCounter++;
                        }
                        logSchedulerI << ss.str() << std::endl;
                    }
                }

                // Prepending the metadata to the scheduler file
                std::ifstream fileIn(controllerFileName); // Open the file for reading
                std::stringstream data;
                data << fileIn.rdbuf(); // Read the file
                std::ofstream controllerFile(controllerFileName); // Open the file for writing (clears the content)
                controllerFile << "#NON-PERMISSIVE" << std::endl << "BEGIN " << obsInfoSize << " 1" << std::endl << data.str(); // Write the data to the file
                logSchedulerI.close();
                controllerFile.close();

                // Run dtcontrol on the generated controller file
                std::string command = "source ./venv/bin/activate && dtcontrol --input " + controllerFileName + " --output stdout:dot";
                STORM_PRINT("Running command: " << command);
                if (std::system(command.c_str()) != 0) {
                    std::cerr << "Failed to run dtcontrol on file. Is it installed? " << controllerFileName << std::endl;
                }
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
//        STORM_PRINT("Observations: " << observations);
//        STORM_PRINT("Observations after switch: " << observationsAfterSwitch);
//        STORM_PRINT("Switch observations: " << switchObservations);
//        STORM_PRINT("Actions: " << actions);
        for (uint64_t obs = 0; obs < observations.size(); ++obs) {
//            if (observations.get(obs) || observationsAfterSwitch.get(obs)) {
//                  STORM_LOG_INFO("For observation: " << obs);
//                auto obsInfo = obsValuations.getStateInfo(obs);
//                STORM_LOG_INFO("For Observation " << obsInfo <<  " with Storm internal id = " << obs << ":");
//            }
//            else {
//                STORM_LOG_INFO("For observation: " << obs << " (not relevant)");
//            }
            if (observations.get(obs)) {
                auto obsInfo = obsValuations.getStateInfo(obs);
//                STORM_LOG_INFO("For Observation " << obsInfo <<  " with Storm internal id = " << obs << ":");
                std::stringstream ss;
                ss << "Observation = " << obsInfo <<  " | Storm internal id = " << obs << " | actions = ";
                auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
                    for (auto act : actions[obs]) {
//                    ss << " " << act;
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

    void exportObservationBasedSchedulers(const storage::sparse::StateValuations& obsValuations, const models::sparse::ChoiceLabeling& choiceLabelling, const std::vector<uint_fast64_t>& choiceIndices,  const std::vector<std::vector<uint64_t>>& statesPerObservation, storm::storage::BitVector const& observations, storm::storage::BitVector const& observationsAfterSwitch) const {
        for (uint64_t obs = 0; obs < observations.size(); ++obs) {
            if (observations.get(obs)) {
                auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
                std::stringstream ss;
                ss << std::endl << " | Storm internal id = " << obs ;

                ss << " | Observation = " ;

                for (const auto& [obsName, obsVal] : obsInfo) {
                    ss << obsName << ": " << obsVal << std::endl;
                }
                ss << " | actions = ";
                auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
                for (auto act : actions[obs]) {
                    //                    ss << " " << act;
                    uint_fast64_t rowIndex = choiceIndices[stateId] + act;
                    auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
                    for (const auto& choiceLabel : choiceLabels) {
                        ss << " " << choiceLabel;
                    }
                }
                ss << " (IsSwitchObservations(" << obs <<")"<<  switchObservations.get(obs)  << ") ";
                if (switchObservations.get(obs)) {
                    ss << " and switch.";
                }
                STORM_LOG_INFO(ss.str());
            }
            if (observationsAfterSwitch.get(obs)) {
                auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
                std::stringstream ss;
                ss << std::endl << " | Storm internal id = " << obs ;

                ss << " | Observation After Switch = " ;

                for (const auto& [obsName, obsVal] : obsInfo) {
                    ss << obsName << ": " << obsVal << std::endl;
                }
                STORM_LOG_INFO(ss.str() << "scheduler ref: " << schedulerRef[obs]);
            }
        }
    }
//
//    void exportObservationBasedSchedulersinFiles(const storage::sparse::StateValuations& obsValuations, const models::sparse::ChoiceLabeling& choiceLabelling, const std::vector<uint_fast64_t>& choiceIndices,  const std::vector<std::vector<uint64_t>>& statesPerObservation, storm::storage::BitVector const& observations, storm::storage::BitVector const& observationsAfterSwitch, uint64_t hash, uint64_t schedulerId) const {
//        std::ofstream logFileObs(std::to_string(hash) + "_scheduler_switch0_memory" +  std::to_string(schedulerId) + ".scheduler"); // switch0 means before switch
//        std::ofstream logFileSwitch(std::to_string(hash) + "_scheduler_switch1_memory" + std::to_string(schedulerId) + ".scheduler"); // switch1 means after switch
//        std::ofstream logFileMemNoSwitch(std::to_string(hash) + "_mem_fun_" +  std::to_string(schedulerId) + ".fsc"); // switch0 means before switch
//        std::ofstream logFileMemSwitch(std::to_string(hash) + "_mem_fun_" + std::to_string(schedulerId) + ".fsc");
//        if (!logFileObs.is_open() || !logFileSwitch.is_open()) {
//            std::cerr << "Failed to open scheduler files" << std::endl;
//            return;
//        }
//        if (!logFileMemNoSwitch.is_open() || !logFileMemSwitch.is_open()) {
//            std::cerr << "Failed to open memory function files" << std::endl;
//            return;
//        }
//
//        for (uint64_t obs = 0; obs < observations.size(); ++obs) {
//            std::stringstream ssNoSwitch;
//            std::stringstream ssWithSwitch;
//            std::stringstream ssMemFunNoSwitch;
//            std::stringstream ssMemFunWithSwitch;
//
//
//            if (observations.get(obs)) {
//                STORM_PRINT("Current considered observation: " << obs);
//                auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
//                for (const auto& [obsName, obsVal] : obsInfo) {
//                    ssNoSwitch << obsVal << " ";
//                }
//
//                auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
//                for (auto act : actions[obs]) {
//                    uint_fast64_t rowIndex = choiceIndices[stateId] + act;
//                    auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
//                    for (const auto& choiceLabel : choiceLabels) {
//                        ssNoSwitch << choiceLabel;
//                    }
//                }
//                ssNoSwitch << std::endl;
//                logFileObs << ssNoSwitch.str(); // Write to observation log file
//                logFileMemNoSwitch << ssMemFunNoSwitch.str() << schedulerRef[obs] << "->" << schedulerRef[obs] << std::endl;
//            }
//
//            if (observationsAfterSwitch.get(obs)) {
//                auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
//                for (const auto& [obsName, obsVal] : obsInfo) {
//                    ssWithSwitch << obsVal << " ";
//                    // ssMemFunWithSwitch << obsVal << " ";
//                }
//                logFileSwitch << ssWithSwitch.str() << "0000000009SCH" << schedulerRef[obs] << std::endl;
//                // logFileMemSwitch << ssMemFunWithSwitch.str() << schedulerRef[obs] << "->" << schedulerId << std::endl;
//            }
//        }
//
//        // Close the file streams
//        logFileObs.close();
//        logFileSwitch.close();
//        logFileMemNoSwitch.close();
//        logFileSwitch.close();
//    }


//

    void exportObservationBasedSchedulersinFiles(const storage::sparse::StateValuations& obsValuations, const models::sparse::ChoiceLabeling& choiceLabelling, const std::vector<uint_fast64_t>& choiceIndices,  const std::vector<std::vector<uint64_t>>& statesPerObservation, storm::storage::BitVector const& observations, storm::storage::BitVector const& observationsAfterSwitch, uint64_t hash, uint64_t schedulerId) const {
        std::ofstream logFileObs(std::to_string(hash) + "_scheduler_switch0_memory" +  std::to_string(schedulerId) + ".scheduler"); // switch0 means before switch
        std::ofstream logFileSwitch(std::to_string(hash) + "_scheduler_switch1_memory" +  std::to_string(schedulerId) + ".scheduler"); // switch1 means after switch
        std::ofstream logFileMemNoSwitch(std::to_string(hash) + "_mem_fun_" +  std::to_string(schedulerId) + ".fsc"); // switch0 means before switch
        std::ofstream logFileMemSwitch(std::to_string(hash) + "_mem_fun_" + std::to_string(schedulerId) + ".fsc");
        std::ofstream mappingFile(std::to_string(hash) + "_action_mapping_ "+ std::to_string(schedulerId) + ".txt");

        if (!logFileObs.is_open() || !logFileSwitch.is_open()) {
            std::cerr << "Failed to open scheduler files" << std::endl;
            return;
        }
        if (!logFileMemNoSwitch.is_open() || !logFileMemSwitch.is_open()) {
            std::cerr << "Failed to open memory function files" << std::endl;
            return;
        }
        if (!mappingFile.is_open()) {
            std::cerr << "Failed to open mapping file" << std::endl;
            return;
        }

        std::map<std::string, int> actionMapping;
        int actionCounter = 0;

        for (uint64_t obs = 0; obs < observations.size(); ++obs) {
            std::stringstream ssNoSwitch;
            std::stringstream ssWithSwitch;
            std::stringstream ssWithAndNoSwitch;
            std::stringstream ssMemFunNoSwitch;
            std::stringstream ssMemFunWithSwitch;

            if (observations.get(obs)) {
                // STORM_PRINT("Current considered observation: " << obs);
                auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
                if (!switchObservations.get(obs)){
                    for (const auto& [obsName, obsVal] : obsInfo) {
                        ssWithAndNoSwitch << obsVal << " ";
                        ssNoSwitch << obsVal << " ";
                    }

                    auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
                    for (auto act : actions[obs]) {
                        uint_fast64_t rowIndex = choiceIndices[stateId] + act;
                        auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
                        for (const auto& choiceLabel : choiceLabels) {
                            if (actionMapping.find(choiceLabel) == actionMapping.end()) {
                                actionMapping[choiceLabel] = actionCounter++;
                            }
                            int actionNumber = actionMapping[choiceLabel];
                            ssWithAndNoSwitch << actionNumber;
                            ssNoSwitch << actionNumber;
                        }
                    }

                    ssWithAndNoSwitch << std::endl;
                    ssNoSwitch << std::endl;

                    logFileObs << ssNoSwitch.str(); // Write to observation file

                    // including the ss string to the file for switch as well
                    logFileSwitch << ssWithAndNoSwitch.str();
                }
                else {
                    for (const auto& [obsName, obsVal] : obsInfo) {
                        ssNoSwitch << obsVal << " ";
                    }
                    auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
                    for (auto act : actions[obs]) {
                        uint_fast64_t rowIndex = choiceIndices[stateId] + act;
                        auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
                        for (const auto& choiceLabel : choiceLabels) {
                            if (actionMapping.find(choiceLabel) == actionMapping.end()) {
                                actionMapping[choiceLabel] = actionCounter++;
                            }
                            int actionNumber = actionMapping[choiceLabel];
                            ssNoSwitch << actionNumber;
                        }
                    }
                    ssNoSwitch << std::endl;
                    std::string noSwitchStr = ssNoSwitch.str(); // Store the generated string for reuse
                    logFileObs << noSwitchStr; // Write to observation file
                    logFileMemNoSwitch << ssMemFunNoSwitch.str() << schedulerRef[obs] << "->" << schedulerRef[obs] << std::endl;
                }
            }

            if (observationsAfterSwitch.get(obs)) {
                auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
                for (const auto& [obsName, obsVal] : obsInfo) {
                    ssWithSwitch << obsVal << " ";
                }
                logFileSwitch << ssWithSwitch.str() << "0000000009SCH" << schedulerRef[obs] << std::endl;
            }
        }

        // Export action mappings to the file
        for (const auto& [actionName, actionNumber] : actionMapping) {
            mappingFile << actionName << " " << actionNumber << std::endl;
        }

        // Close the file streams
        logFileObs.close();
        logFileSwitch.close();
        logFileMemNoSwitch.close();
        logFileMemSwitch.close();
        mappingFile.close();
    }

    ObservationSchedulerMoore update_fsc_moore(const models::sparse::ChoiceLabeling& choiceLabelling, const std::vector<uint_fast64_t>& choiceIndices,  const std::vector<std::vector<uint64_t>>& statesPerObservation, storm::storage::BitVector const& observations, storm::storage::BitVector const& observationsAfterSwitch, std::unordered_map<uint64_t, uint64_t> winningObservationsFirstScheduler ,ObservationSchedulerMoore schedulerMoore, uint64_t schedulerId) const {
        int primeMemoryOffset = 1000000000;
        bool isSwitch = false;

        for (uint64_t obs = 0; obs < observations.size(); ++obs) {
            std::vector<std::string> actionVector;
            if (observations.get(obs)) {
                if (!switchObservations.get(obs)){
                    auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
                    for (auto act : actions[obs]) {
                        uint_fast64_t rowIndex = choiceIndices[stateId] + act;
                        auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
                        for (const auto& choiceLabel : choiceLabels) {
                            actionVector.push_back(choiceLabel);
                        }
                    }
                    schedulerMoore.nextMemoryTransition[schedulerId][obs] = schedulerId;
                    schedulerMoore.actionSelection[schedulerId][obs] = actionVector;
                }
                else {
                    isSwitch = true;
                    auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
                    for (auto act : actions[obs]) {
                        uint_fast64_t rowIndex = choiceIndices[stateId] + act;
                        auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
                        for (const auto& choiceLabel : choiceLabels) {
                            actionVector.push_back(choiceLabel);
                        }
                    }
                    schedulerMoore.nextMemoryTransition[schedulerId][obs] = primeMemoryOffset + schedulerId;
                    schedulerMoore.actionSelection[primeMemoryOffset+schedulerId][obs] = actionVector;
                }
            }

            if (observationsAfterSwitch.get(obs)) {
                schedulerMoore.nextMemoryTransition[schedulerId][obs] = schedulerRef[obs];
            }
            if (winningObservationsFirstScheduler.find(obs) != winningObservationsFirstScheduler.end()) {
                schedulerMoore.nextMemoryTransition[schedulerId][obs] = winningObservationsFirstScheduler[obs];
            }
        }

        if(isSwitch){
            // add or replicate transitions from the `primed-memory` function and action selection
            for (uint64_t obs = 0; obs < observations.size(); ++obs) {
                // add or replicate transitions from the `primed-memory` function
                if (schedulerMoore.nextMemoryTransition[primeMemoryOffset + schedulerId].find(obs) == schedulerMoore.nextMemoryTransition[primeMemoryOffset + schedulerId].end()){
                    // todo: in case of switch self loop of non-switch memory should be copied to the switch memory
                    schedulerMoore.nextMemoryTransition[primeMemoryOffset + schedulerId][obs] = schedulerMoore.nextMemoryTransition[schedulerId][obs];
                }
                // replicate action selections for each memory and observation
                if (schedulerMoore.actionSelection[primeMemoryOffset + schedulerId].find(obs) == schedulerMoore.actionSelection[primeMemoryOffset + schedulerId].end()){
                    schedulerMoore.actionSelection[primeMemoryOffset + schedulerId][obs] = schedulerMoore.actionSelection[schedulerId][obs];
                }
            }
        }

        schedulerMoore.initialNode = schedulerId;
        return schedulerMoore;
    }

//
//    void exportObservationBasedSchedulersinFiles(const storage::sparse::StateValuations& obsValuations, const models::sparse::ChoiceLabeling& choiceLabelling, const std::vector<uint_fast64_t>& choiceIndices,  const std::vector<std::vector<uint64_t>>& statesPerObservation, storm::storage::BitVector const& observations, storm::storage::BitVector const& observationsAfterSwitch, uint64_t hash, uint64_t schedulerId) const {
//        std::ofstream logFileObs(std::to_string(hash) + "_scheduler_switch0_memory" +  std::to_string(schedulerId) + ".scheduler"); // switch0 means before switch
//        std::ofstream logFileSwitch(std::to_string(hash) + "_scheduler_switch1_memory" +  std::to_string(schedulerId) + ".scheduler"); // switch1 means after switch
//        std::ofstream logFileMemNoSwitch(std::to_string(hash) + "_mem_fun_" +  std::to_string(schedulerId) + ".fsc"); // switch0 means before switch
//        std::ofstream logFileMemSwitch(std::to_string(hash) + "_mem_fun_" + std::to_string(schedulerId) + ".fsc");
//        std::ofstream mappingFile(std::to_string(hash) + "_action_mapping_ "+ std::to_string(schedulerId) + ".txt");
//
//        if (!logFileObs.is_open() || !logFileSwitch.is_open()) {
//            std::cerr << "Failed to open scheduler files" << std::endl;
//            return;
//        }
//        if (!logFileMemNoSwitch.is_open() || !logFileMemSwitch.is_open()) {
//            std::cerr << "Failed to open memory function files" << std::endl;
//            return;
//        }
//        if (!mappingFile.is_open()) {
//            std::cerr << "Failed to open mapping file" << std::endl;
//            return;
//        }
//
//        std::map<std::string, int> actionMapping;
//        int actionCounter = 0;
//
//        for (uint64_t obs = 0; obs < observations.size(); ++obs) {
//            std::stringstream ssNoSwitch;
//            std::stringstream ssWithSwitch;
//            std::stringstream ssWithAndNoSwitch;
//            std::stringstream ssMemFunNoSwitch;
//            std::stringstream ssMemFunWithSwitch;
//
//            if (observations.get(obs)) {
//                // STORM_PRINT("Current considered observation: " << obs);
//                auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
//                if (!switchObservations.get(obs)){
//                    for (const auto& [obsName, obsVal] : obsInfo) {
//                        ssWithAndNoSwitch << obsVal << " ";
//                        ssNoSwitch << obsVal << " ";
//                    }
//
//                    auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
//                    for (auto act : actions[obs]) {
//                        uint_fast64_t rowIndex = choiceIndices[stateId] + act;
//                        auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
//                        for (const auto& choiceLabel : choiceLabels) {
//                            if (actionMapping.find(choiceLabel) == actionMapping.end()) {
//                                actionMapping[choiceLabel] = actionCounter++;
//                            }
//                            int actionNumber = actionMapping[choiceLabel];
//                            ssWithAndNoSwitch << actionNumber;
//                            ssNoSwitch << actionNumber;
//                        }
//                    }
//
//                    ssWithAndNoSwitch << std::endl;
//                    ssNoSwitch << std::endl;
//
//                    logFileObs << ssNoSwitch.str(); // Write to observation file
//
//                    // including the ss string to the file for switch as well
//                    logFileSwitch << ssWithAndNoSwitch.str();
//                }
//                else {
//                    for (const auto& [obsName, obsVal] : obsInfo) {
//                        ssNoSwitch << obsVal << " ";
//                    }
//                    auto stateId = statesPerObservation[obs][0]; // assuming it is enough to look at the first state to get the correct action
//                    for (auto act : actions[obs]) {
//                        uint_fast64_t rowIndex = choiceIndices[stateId] + act;
//                        auto choiceLabels = choiceLabelling.getLabelsOfChoice(rowIndex);
//                        for (const auto& choiceLabel : choiceLabels) {
//                            if (actionMapping.find(choiceLabel) == actionMapping.end()) {
//                                actionMapping[choiceLabel] = actionCounter++;
//                            }
//                            int actionNumber = actionMapping[choiceLabel];
//                            ssNoSwitch << actionNumber;
//                        }
//                    }
//                    ssNoSwitch << std::endl;
//                    std::string noSwitchStr = ssNoSwitch.str(); // Store the generated string for reuse
//                    logFileObs << noSwitchStr; // Write to observation file
//                    logFileMemNoSwitch << ssMemFunNoSwitch.str() << schedulerRef[obs] << "->" << schedulerRef[obs] << std::endl;
//                }
//            }
//
//            if (observationsAfterSwitch.get(obs)) {
//                auto obsInfo = obsValuations.getObsevationValuationforExplainability(obs);
//                for (const auto& [obsName, obsVal] : obsInfo) {
//                    ssWithSwitch << obsVal << " ";
//                }
//                logFileSwitch << ssWithSwitch.str() << "0000000009SCH" << schedulerRef[obs] << std::endl;
//            }
//        }
//
//        // Export action mappings to the file
//        for (const auto& [actionName, actionNumber] : actionMapping) {
//            mappingFile << actionName << " " << actionNumber << std::endl;
//        }
//
//        // Close the file streams
//        logFileObs.close();
//        logFileSwitch.close();
//        logFileMemNoSwitch.close();
//        logFileMemSwitch.close();
//        mappingFile.close();
//    }
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
