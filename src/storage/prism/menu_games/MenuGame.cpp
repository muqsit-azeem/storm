#include "src/storage/prism/menu_games/MenuGame.h"

#include "src/exceptions/InvalidOperationException.h"
#include "src/exceptions/InvalidArgumentException.h"

#include "src/storage/dd/CuddBdd.h"
#include "src/storage/dd/CuddAdd.h"

#include "src/models/symbolic/StandardRewardModel.h"

namespace storm {
    namespace prism {
        namespace menu_games {
            
            template<storm::dd::DdType Type>
            MenuGame<Type>::MenuGame(std::shared_ptr<storm::dd::DdManager<Type>> manager,
                                     storm::dd::Bdd<Type> reachableStates,
                                     storm::dd::Bdd<Type> initialStates,
                                     storm::dd::Add<Type> transitionMatrix,
                                     std::set<storm::expressions::Variable> const& rowVariables,
                                     std::set<storm::expressions::Variable> const& columnVariables,
                                     std::vector<std::pair<storm::expressions::Variable, storm::expressions::Variable>> const& rowColumnMetaVariablePairs,
                                     std::set<storm::expressions::Variable> const& player1Variables,
                                     std::set<storm::expressions::Variable> const& player2Variables,
                                     std::set<storm::expressions::Variable> const& allNondeterminismVariables,
                                     storm::expressions::Variable const& updateVariable,
                                     std::map<storm::expressions::Expression, storm::dd::Bdd<Type>> const& expressionToBddMap) : storm::models::symbolic::StochasticTwoPlayerGame<Type>(manager, reachableStates, initialStates, transitionMatrix.sumAbstract({updateVariable}), rowVariables, nullptr, columnVariables, nullptr, rowColumnMetaVariablePairs, player1Variables, player2Variables, allNondeterminismVariables), updateVariable(updateVariable), expressionToBddMap(expressionToBddMap) {
                // Intentionally left empty.
            }
            
            template<storm::dd::DdType Type>
            storm::dd::Bdd<Type> MenuGame<Type>::getStates(std::string const& label) const {
                STORM_LOG_THROW(false, storm::exceptions::InvalidOperationException, "Menu games do not provide labels.");
            }
            
            template<storm::dd::DdType Type>
            storm::dd::Bdd<Type> MenuGame<Type>::getStates(storm::expressions::Expression const& expression) const {
                return this->getStates(expression, false);
            }
            
            template<storm::dd::DdType Type>
            storm::dd::Bdd<Type> MenuGame<Type>::getStates(storm::expressions::Expression const& expression, bool negated) const {
                auto it = expressionToBddMap.find(expression);
                STORM_LOG_THROW(it != expressionToBddMap.end(), storm::exceptions::InvalidArgumentException, "The given expression was not used in the abstraction process and can therefore not be retrieved.");
                if (negated) {
                    return !it->second && this->getReachableStates();
                } else {
                    return it->second && this->getReachableStates();
                }
            }
            
            template<storm::dd::DdType Type>
            bool MenuGame<Type>::hasLabel(std::string const& label) const {
                return false;
            }
            
            template class MenuGame<storm::dd::DdType::CUDD>;
            
        } // namespace menu_games
    } // namespace prism
} // namespace storm
