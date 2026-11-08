"""Small pronoun-resolution dataset used by 23-transformer.py."""

pets = ["cat", "dog", "rabbit", "hamster", "parrot", "turtle", "fish", "horse"]

_train_examples = [
    ("the cat found a box and it slept", "cat"),
    ("the cat ate its food and it purred", "cat"),
    ("the cat sat by the window and it watched birds", "cat"),
    ("the dog chased a ball and it returned", "dog"),
    ("the dog heard a noise and it barked", "dog"),
    ("the dog went for a walk and it enjoyed the park", "dog"),
    ("the rabbit nibbled a carrot and it hopped away", "rabbit"),
    ("the rabbit entered its burrow and it rested", "rabbit"),
    ("the rabbit saw a shadow and it ran quickly", "rabbit"),
    ("the hamster ran in its wheel and it got tired", "hamster"),
    ("the hamster stored some seeds and it ate later", "hamster"),
    ("the hamster woke at night and it explored its cage", "hamster"),
    ("the parrot learned a phrase and it repeated it", "parrot"),
    ("the parrot sat on its perch and it sang", "parrot"),
    ("the parrot saw its owner and it squawked", "parrot"),
    ("the turtle crossed the garden and it moved slowly", "turtle"),
    ("the turtle hid in its shell and it stayed safe", "turtle"),
    ("the turtle reached the water and it swam", "turtle"),
    ("the fish swam around the tank and it found food", "fish"),
    ("the fish hid behind a plant and it stayed still", "fish"),
    ("the fish saw a bubble and it swam upward", "fish"),
    ("the horse ran across the field and it stopped", "horse"),
    ("the horse ate some hay and it looked content", "horse"),
    ("the horse heard its name and it came closer", "horse"),
]

_test_examples = [
    ("the cat climbed a tree and it looked down", "cat"),
    ("the dog found its toy and it played outside", "dog"),
    ("the rabbit left its nest and it searched for clover", "rabbit"),
    ("the hamster found a tunnel and it crawled through", "hamster"),
    ("the parrot heard a tune and it began to sing", "parrot"),
    ("the turtle found a sunny spot and it rested there", "turtle"),
    ("the fish saw its keeper and it swam to the glass", "fish"),
    ("the horse crossed a stream and it drank water", "horse"),
]

train = [(sentence.split(), pets.index(pet)) for sentence, pet in _train_examples]
test = [(sentence.split(), pets.index(pet)) for sentence, pet in _test_examples]
words = sorted({word for sentence, _ in _train_examples + _test_examples for word in sentence.split()})
