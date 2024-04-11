# bobnet
BobNet is a modular, portable Large Language Model built in community using consumer-grade hardware. The framework ingests training data and emits sharable "Bob" files that represent itty-bitty >5M Tensorflow language models trained only on one brief text each (represented by a training data file). The same models that are emitted are stored in a local SqliteDB as a vector store.

When you do inference, a vector search is done with your query first, the most appropriate models are retrieved and then the highest-probability result from each round is selected as the next token. It feels like in #ai they are always coming up with fancy new terms I don't understand, so I propose "mixture of Bob" or M.O.B. for short as the name of this design.

The goal is to crowd-source training of individual topics into Bob files that can be shared with others. Because Bob files can be emitted, shared and then ingested elsewhere, you can build a BobNet specific to your use case, and it will only be aware of what you provide. Fully modular.

BobNet was created to try and address the following problems I see in the current LLM ecosystem:

1. Capable models can only currently be built at great expense using specialized hardware.
2. A model built in this way is a monolith, where you have to either take it or leave it in its entirety.
3. A model built this way contain training data that is not controlled by the consumer.
4. Communities and individuals must rely on the good favor of large, for-profit corporations rather than building something themselves.

BobNet is built and tested using CPU-only on consumer-grade hardware (currently an HP Z640). Rather than forcing users to acquire more and more VRAM to execute a model, BobNet has a very small resource footprint, relying on storage space as its most limiting resource (cheep!). Every individual .bob file trained on a text is shareable / portable, and a BobNet can be built selectively at the discretion of the user. This means you can include just the contents you want, like general conversation, specialized information for your organization, general facts, etc... but opt-in rather than relying on prompts to protect your users from uninformed responses. Finally, BobNet can be built

Help build the BobNet! Join the revolution!

FUTURE WORK
1. "Pet Store" - interface between BobNet instances to allow truly distributed, specialized inference
    1. Ability to either interface locally (for example, in a Raspberry Pi cluster on a LAN)
    2. Ability to interface across networks
2. General optimization and improvement
    1. It takes 5 minutes per 256 char of text currently to train on a single core of an Intel Xeon E5-2690
    2. It takes ~50MB of storage space per 256 char of text when persisted to a .bob file

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
    4. Models will be penalized to the degree to which they are unfamiliar with any part of the question text.
5. You can share *.bob files with other people
    1. Each file represents the work output of training on a single text input file
    2. You can import *.bob files shared by others by putting them in the "import" subdirectory
    3. As a result, you can build your BobNet in a modular fashion, only including approved sources