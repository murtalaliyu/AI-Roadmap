"""Sample service agreement text and retrieval questions for rag.py."""

pieces = [
    "Payment: The client must pay each invoice within 30 days of its issue date.",
    "Term: This agreement begins on the effective date and continues for one year.",
    "Renewal: The agreement renews for additional one-year terms unless either party gives 30 days' written notice before the current term ends.",
    "Termination: Either party may terminate this agreement for convenience with 30 days' written notice.",
    "Confidentiality: Each party must protect the other party's confidential information and use it only to perform this agreement.",
    "Intellectual property: The client owns the final work product after paying all fees due. The provider retains ownership of its pre-existing materials.",
    "Liability: Neither party is liable for indirect or consequential damages. Each party's total liability is limited to the fees paid in the 12 months before the claim.",
    "Governing law: The laws of the state where the client is headquartered govern this agreement.",
]

questions = [
    ("What is the payment deadline for each invoice?", 0),
    ("How long does the initial agreement last?", 1),
    ("How does the agreement renew for additional terms?", 2),
    ("How can either party end the agreement?", 3),
    ("How must confidential information be handled?", 4),
    ("Who owns the final work product?", 5),
    ("What is the limit on liability?", 6),
    ("Which law governs the agreement?", 7),
]

common = {
    "a", "an", "and", "are", "as", "at", "be", "before", "by", "for",
    "from", "how", "if", "in", "is", "it", "must", "of", "on", "or",
    "the", "this", "to", "what", "when", "where", "which", "who", "with",
}
