# bobnet
A modular, portable Large Language Model built in community using consumer-grade hardware

## The Story of Bob

This is Bob:

![alt text](images/bob.png)

Bob isn't very strong on his own:

![alt text](images/bob_not_strong.png)

But when he and his friends get together, they can do great things:

![alt text](images/bob_together.png)

Also, Bob is portable - you can share him with your friends!

![alt text](images/bob_shared.png)

## Getting Started

Take the following steps to start using BobNet:

1. Clone this repo and run "install.bat".
2. Copy text training data under the "ingest" subfolder
    1. Each individual file will become its own "Bob" language model
    2. Smaller, focused files are best.
3. To train, run "run_bob_net.bat" with no arguments
    1. BobNet will ingest the training data you provided.
    2. It will store a language model per training file in a vector store.
    3. It will also emit a *.bob file in the "share" subdirectory that you can share with others.
4. To do inference, run "run_bob_net.bat" with one argument, the question you are trying to answer
    1. Example: "What is 2 + 2?"
    2. BobNet will do a vector search to find which internal language models best fit your question.
    3. BobNet will use each identified model to do inference, providing only the most confident result.
    4. Models will be penalized accordingly to avoid results that are overconfident but way wrong.
5. You can share *.bob files with other people
    1. Each file represents the work output of training on a single text input file
    2. You can import *.bob files shared by others by putting them in the "import" subdirectory
    3. As a result, you can build your BobNet in a modular fashion, only including approved sources