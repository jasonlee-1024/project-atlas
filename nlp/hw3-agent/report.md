# Report

## Part 3

### 1. Agent and Pass@1 Score

I chose the **LangChain (LangGraph) agent** (`tool-calling-langchain`). Its Pass@1 score is **50%** (5/10 tasks passed).

---

### 2. Failure Case Analysis

1. **Task 0** failed because the number of steps reached the maximum limit. The user simulator did not provide the order ID upfront, causing the agent to call `get_user_details` to retrieve the order list. This extra round trip consumed additional steps and the conversation hit `max_num_steps=30` before `exchange_delivered_order_items` could be called.

2. **Task 5** failed because the agent used `gift_card` as the payment method while the expected payment was `paypal`. However, this task is arguably a success — the prompt does not specify a payment method, and when the agent asked the user, the simulator explicitly responded *"I'd prefer using the gift card."* The agent correctly followed the user's stated preference, making this a user simulator inconsistency rather than an agent error.

3. **Task 6** failed because the agent used the wrong `item_ids` when calling `return_delivered_order_items`. The root cause is the user simulator, which gave an invalid prompt: *"Actually, I want to return the watch from the pending order, not the delivered order"* — a logically impossible request since both pending orders had already been cancelled and their items are ineligible for return. This confused the agent and derailed the return flow entirely.

---

### 3. Suggested Fixes

1. **Task 0**: Increase `max_num_steps` to give the agent more room to complete multi-turn tasks when the user simulator takes unexpected detours.
2. **Task 5 & 6**: Give more specific prompts to the user simulator so that it selects the correct payment method and avoids issuing logically contradictory requests.
