#ifndef STORM_LOGIC_FORMULA_H_
#define STORM_LOGIC_FORMULA_H_

#include <memory>
#include <vector>
#include <iostream>
#include <set>

#include <boost/any.hpp>

#include "src/storage/expressions/Variable.h"
#include "src/storage/expressions/Expression.h"

#include "src/logic/FormulasForwardDeclarations.h"

namespace storm {
    namespace logic {

        // Forward-declare visitor for accept() method.
        class FormulaVisitor;

        // Also foward-declare base model checker class.
        class ModelChecker;
        
        class Formula : public std::enable_shared_from_this<Formula const> {
        public:
            // Make the destructor virtual to allow deletion of objects of subclasses via a pointer to this class.
            virtual ~Formula() {
                // Intentionally left empty.
            };
            
            friend std::ostream& operator<<(std::ostream& out, Formula const& formula);

            // Basic formula types.
            virtual bool isPathFormula() const;
            virtual bool isStateFormula() const;
            virtual bool isConditionalProbabilityFormula() const;
            virtual bool isConditionalRewardFormula() const;
            
            virtual bool isProbabilityPathFormula() const;
            virtual bool isRewardPathFormula() const;
            virtual bool isExpectedTimePathFormula() const;

            virtual bool isBinaryBooleanStateFormula() const;
            virtual bool isUnaryBooleanStateFormula() const;

            // Operator formulas.
            virtual bool isOperatorFormula() const;
            virtual bool isLongRunAverageOperatorFormula() const;
            virtual bool isExpectedTimeOperatorFormula() const;
            virtual bool isProbabilityOperatorFormula() const;
            virtual bool isRewardOperatorFormula() const;

            // Atomic state formulas.
            virtual bool isBooleanLiteralFormula() const;
            virtual bool isTrueFormula() const;
            virtual bool isFalseFormula() const;
            virtual bool isAtomicExpressionFormula() const;
            virtual bool isAtomicLabelFormula() const;

            // Probability path formulas.
            virtual bool isNextFormula() const;
            virtual bool isUntilFormula() const;
            virtual bool isBoundedUntilFormula() const;
            virtual bool isEventuallyFormula() const;
            virtual bool isGloballyFormula() const;

            // Reward formulas.
            virtual bool isCumulativeRewardFormula() const;
            virtual bool isInstantaneousRewardFormula() const;
            virtual bool isReachabilityRewardFormula() const;
            virtual bool isLongRunAverageRewardFormula() const;
            
            // Expected time formulas.
            virtual bool isReachbilityExpectedTimeFormula() const;
            
            // Type checks for abstract intermediate classes.
            virtual bool isBinaryPathFormula() const;
            virtual bool isBinaryStateFormula() const;
            virtual bool isUnaryPathFormula() const;
            virtual bool isUnaryStateFormula() const;

            virtual boost::any accept(FormulaVisitor const& visitor, boost::any const& data) const = 0;
            
            static std::shared_ptr<Formula const> getTrueFormula();
            
            PathFormula& asPathFormula();
            PathFormula const& asPathFormula() const;
        
            StateFormula& asStateFormula();
            StateFormula const& asStateFormula() const;
            
            BinaryStateFormula& asBinaryStateFormula();
            BinaryStateFormula const& asBinaryStateFormula() const;
            
            UnaryStateFormula& asUnaryStateFormula();
            UnaryStateFormula const& asUnaryStateFormula() const;
            
            BinaryBooleanStateFormula& asBinaryBooleanStateFormula();
            BinaryBooleanStateFormula const& asBinaryBooleanStateFormula() const;

            UnaryBooleanStateFormula& asUnaryBooleanStateFormula();
            UnaryBooleanStateFormula const& asUnaryBooleanStateFormula() const;

            BooleanLiteralFormula& asBooleanLiteralFormula();
            BooleanLiteralFormula const& asBooleanLiteralFormula() const;
            
            AtomicExpressionFormula& asAtomicExpressionFormula();
            AtomicExpressionFormula const& asAtomicExpressionFormula() const;
            
            AtomicLabelFormula& asAtomicLabelFormula();
            AtomicLabelFormula const& asAtomicLabelFormula() const;
            
            UntilFormula& asUntilFormula();
            UntilFormula const& asUntilFormula() const;
            
            BoundedUntilFormula& asBoundedUntilFormula();
            BoundedUntilFormula const& asBoundedUntilFormula() const;
            
            EventuallyFormula& asEventuallyFormula();
            EventuallyFormula const& asEventuallyFormula() const;
            
            GloballyFormula& asGloballyFormula();
            GloballyFormula const& asGloballyFormula() const;
            
            BinaryPathFormula& asBinaryPathFormula();
            BinaryPathFormula const& asBinaryPathFormula() const;
            
            UnaryPathFormula& asUnaryPathFormula();
            UnaryPathFormula const& asUnaryPathFormula() const;
            
            ConditionalFormula& asConditionalFormula();
            ConditionalFormula const& asConditionalFormula() const;
            
            NextFormula& asNextFormula();
            NextFormula const& asNextFormula() const;
            
            LongRunAverageOperatorFormula& asLongRunAverageOperatorFormula();
            LongRunAverageOperatorFormula const& asLongRunAverageOperatorFormula() const;

            ExpectedTimeOperatorFormula& asExpectedTimeOperatorFormula();
            ExpectedTimeOperatorFormula const& asExpectedTimeOperatorFormula() const;
            
            CumulativeRewardFormula& asCumulativeRewardFormula();
            CumulativeRewardFormula const& asCumulativeRewardFormula() const;
            
            InstantaneousRewardFormula& asInstantaneousRewardFormula();
            InstantaneousRewardFormula const& asInstantaneousRewardFormula() const;
            
            LongRunAverageRewardFormula& asLongRunAverageRewardFormula();
            LongRunAverageRewardFormula const& asLongRunAverageRewardFormula() const;
            
            ProbabilityOperatorFormula& asProbabilityOperatorFormula();
            ProbabilityOperatorFormula const& asProbabilityOperatorFormula() const;
            
            RewardOperatorFormula& asRewardOperatorFormula();
            RewardOperatorFormula const& asRewardOperatorFormula() const;
            
            OperatorFormula& asOperatorFormula();
            OperatorFormula const& asOperatorFormula() const;
            
            std::vector<std::shared_ptr<AtomicExpressionFormula const>> getAtomicExpressionFormulas() const;
            std::vector<std::shared_ptr<AtomicLabelFormula const>> getAtomicLabelFormulas() const;
            std::set<std::string> getReferencedRewardModels() const;
            
            std::shared_ptr<Formula const> asSharedPointer();
            std::shared_ptr<Formula const> asSharedPointer() const;
            
            virtual std::shared_ptr<Formula> substitute(std::map<storm::expressions::Variable, storm::expressions::Expression> const& substitution) const = 0;
            
            std::string toString() const;
            virtual std::ostream& writeToStream(std::ostream& out) const = 0;
            
            virtual void gatherAtomicExpressionFormulas(std::vector<std::shared_ptr<AtomicExpressionFormula const>>& atomicExpressionFormulas) const;
            virtual void gatherAtomicLabelFormulas(std::vector<std::shared_ptr<AtomicLabelFormula const>>& atomicExpressionFormulas) const;
            virtual void gatherReferencedRewardModels(std::set<std::string>& referencedRewardModels) const;
            
        private:
            // Currently empty.
        };
        
        std::ostream& operator<<(std::ostream& out, Formula const& formula);
    }
}

#endif /* STORM_LOGIC_FORMULA_H_ */