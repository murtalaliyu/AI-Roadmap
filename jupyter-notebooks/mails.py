from itertools import islice, product


_SPAM_SUBJECTS = (
    "congratulations you are our lucky winner",
    "urgent claim your free reward",
    "exclusive offer just for you",
    "you have been selected for a cash prize",
    "limited time bonus waiting for you",
    "your free vacation is ready to claim",
)

_SPAM_DETAILS = (
    "receive a gift card worth one hundred dollars",
    "get a free phone with no purchase required",
    "earn extra cash from home this week",
    "unlock a special discount on your next order",
    "collect bonus points and redeem them today",
    "claim a luxury trip before this offer expires",
)

_SPAM_ACTIONS = (
    "click the link and enter your details now",
    "reply immediately to confirm your account",
    "visit our website to claim your prize",
    "send your information to receive the payment",
    "act now because this deal will not last",
    "follow the instructions to get your reward",
)

_SPAM_SIGNOFFS = (
    "you are a winner do not miss out",
    "this offer is guaranteed for selected members",
    "congratulations your reward is waiting",
    "apply today and enjoy your free bonus",
    "claim now to avoid losing this opportunity",
    "thank you for choosing our prize service",
)

_HAM_OPENINGS = (
    "hello the team meeting has moved to tomorrow",
    "hi your monthly account statement is ready",
    "good morning I shared the project notes",
    "hello the package was delivered this afternoon",
    "hi please review the updated work schedule",
    "good afternoon your appointment is confirmed",
)

_HAM_DETAILS = (
    "the report includes the latest sales figures",
    "please bring the documents to the office",
    "we can discuss the changes during our call",
    "your order is available for pickup at the store",
    "the class begins at nine on Monday morning",
    "I added the requested files to the shared folder",
)

_HAM_ACTIONS = (
    "let me know if you have any questions",
    "please reply when you have a chance",
    "you can find the details in the attached file",
    "call the office if you need to reschedule",
    "thanks for taking the time to review it",
    "we will send another update next week",
)

_HAM_SIGNOFFS = (
    "best regards from the support team",
    "thank you and have a pleasant day",
    "sincerely the customer service team",
    "see you at the meeting tomorrow",
    "thanks again for your help",
    "kind regards from your project team",
)


def _make_messages(groups, count):
    return [" ".join(parts) for parts in islice(product(*groups), count)]


_spam_messages = _make_messages(
    (_SPAM_SUBJECTS, _SPAM_DETAILS, _SPAM_ACTIONS, _SPAM_SIGNOFFS), 500
)
_spam_messages[0] = "you won a free trip claim it now claim your reward today"
_ham_messages = _make_messages(
    (_HAM_OPENINGS, _HAM_DETAILS, _HAM_ACTIONS, _HAM_SIGNOFFS), 500
)

mails = []
spam = []
for _spam_message, _ham_message in zip(_spam_messages, _ham_messages):
    mails.extend((_spam_message, _ham_message))
    spam.extend((1, 0))
