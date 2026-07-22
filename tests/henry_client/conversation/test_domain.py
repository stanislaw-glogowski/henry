from henry_client.conversation.domain import ConversationStore, MessageRole


def test_conversation_store_keeps_recent_messages() -> None:
    store = ConversationStore()

    for index in range(4):
        store.add_user_message(f"user-{index}")
        store.add_assistant_message(f"assistant-{index}")

    assert len(store.messages) == store._MAX_LEN
    assert store.messages[-1].content == "assistant-3"


def test_conversation_store_assigns_roles_and_resets() -> None:
    store = ConversationStore()
    store.add_user_message("hello")
    store.add_assistant_message("hi")

    assert [message.role for message in store.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]

    store.reset()
    assert store.messages == ()
