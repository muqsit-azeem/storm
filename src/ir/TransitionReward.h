/*
 * TransitionReward.h
 *
 *  Created on: Jan 10, 2013
 *      Author: Christian Dehnert
 */

#ifndef STORM_IR_TRANSITIONREWARD_H_
#define STORM_IR_TRANSITIONREWARD_H_

#include "expressions/BaseExpression.h"

#include <memory>

namespace storm {

namespace ir {

/*!
 * A class representing a transition reward.
 */
class TransitionReward {
public:
	/*!
	 * Default constructor. Creates an empty transition reward.
	 */
	TransitionReward();

	/*!
	 * Creates a transition reward for the transitions with the given name emanating from states
	 * satisfying the given expression with the value given by another expression.
	 * @param commandName the name of the command that obtains this reward.
	 * @param statePredicate the predicate that needs to hold before taking a transition with the
	 * previously specified name in order to obtain the reward.
	 * @param rewardValue an expression specifying the values of the rewards to attach to the
	 * transitions.
	 */
	TransitionReward(std::string commandName, std::shared_ptr<storm::ir::expressions::BaseExpression> statePredicate, std::shared_ptr<storm::ir::expressions::BaseExpression> rewardValue);

	/*!
	 * Retrieves a string representation of this transition reward.
	 * @returns a string representation of this transition reward.
	 */
	std::string toString() const;

	/*!
	 * Retrieves reward for given transition.
	 * Returns reward value if source state fulfills predicate and the transition is labeled correctly, zero otherwise.
	 */
	double getReward(std::string const & label, std::pair<std::vector<bool>, std::vector<int_fast64_t>> const * state) const;

private:
	// The name of the command this transition-based reward is attached to.
	std::string commandName;

	// A predicate that needs to be satisfied by states for the reward to be obtained (by taking
	// a corresponding command transition).
	std::shared_ptr<storm::ir::expressions::BaseExpression> statePredicate;

	// The expression specifying the value of the reward obtained along the transitions.
	std::shared_ptr<storm::ir::expressions::BaseExpression> rewardValue;
};

} // namespace ir

} // namespace storm

#endif /* STORM_IR_TRANSITIONREWARD_H_ */
