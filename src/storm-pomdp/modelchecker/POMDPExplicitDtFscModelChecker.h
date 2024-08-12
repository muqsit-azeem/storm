//
// Created by Muqsit Azeem on 11.08.24.
//
#include "storm/models/sparse/Pomdp.h"
#include "storm-pomdp/analysis/IterativePolicySearch.h"
#include "storm/models/sparse/Dtmc.h"
#include "storm-pomdp/analysis/FormulaInformation.h"
#include "storm/storage/Distribution.h"

namespace storm {
namespace pomdp {
template<typename ValueType>
class PomdpExplicitDtFscModelChecker {
   public:
    explicit PomdpExplicitDtFscModelChecker(storm::models::sparse::Pomdp<ValueType> pomdp, ObservationPolicyPosteriorMealy policy);
    std::pair<uint64_t, storm::storage::Distribution<ValueType, uint64_t>> getSuccessorNodeAndActionDistribution(
        uint64_t currentPomdpState, uint64_t currentPolicyNode);

   private:
    std::shared_ptr<storm::models::sparse::Dtmc<ValueType>> buildInducedMarkovChain(storm::models::sparse::Pomdp<ValueType> const& pomdp, storm::pomdp::analysis::FormulaInformation const& formulaInfo);
    std::shared_ptr<ValueType> inputPomdp;
    ObservationPolicyPosteriorMealy inputPolicy;
    std::set<uint32_t> targetObservations;
};

}  // namespace pomdp
}  // namespace storm

