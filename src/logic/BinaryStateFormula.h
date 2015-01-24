#ifndef STORM_LOGIC_BINARYSTATEFORMULA_H_
#define STORM_LOGIC_BINARYSTATEFORMULA_H_

#include "src/logic/StateFormula.h"

namespace storm {
    namespace logic {
        class BinaryStateFormula : public StateFormula {
        public:
            BinaryStateFormula(std::shared_ptr<Formula const> const& leftSubformula, std::shared_ptr<Formula const> const& rightSubformula);
            
            virtual ~BinaryStateFormula() {
                // Intentionally left empty.
            }
            
            virtual bool isBinaryStateFormula() const override;

            Formula const& getLeftSubformula() const;
            Formula const& getRightSubformula() const;
            
        private:
            std::shared_ptr<Formula const> leftSubformula;
            std::shared_ptr<Formula const> rightSubformula;
        };
    }
}

#endif /* STORM_LOGIC_BINARYSTATEFORMULA_H_ */