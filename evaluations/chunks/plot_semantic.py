from matplotlib import pyplot as plt
import json

if __name__ == '__main__':
    results = json.load(open("eval_semantic.json", "r"))

    chunk_batches = [100, 200, 400, 500, 1000]
    chunks_overlaps = [0.1, 0.2, 0.3, 0.4, 0.5]

    chunkers = list(results.keys())

    fig = plt.figure()

    """ for chunker in chunkers:
        times = []
        for batch in chunk_batches:
            times.append(results[chunker][f"{batch}-0.3"]["time"])
        plt.plot(chunk_batches, times, label=f"{chunker}")

    plt.xlabel('Chunk Batches')
    plt.ylabel('Times [s]')
    plt.title('Times vs Chunk Batches for Different Chunkers (Overlap=30%)')
    plt.grid()
    plt.legend()
    plt.show() """

    """ for chunker in chunkers:
        times = []
        for overlap in chunks_overlaps:
            times.append(results[chunker][f"400-{overlap}"]["time"])
        plt.plot(chunks_overlaps, times, label=f"{chunker}")

    plt.xlabel('Chunk Overlaps')
    plt.ylabel('Times [s]')
    plt.title('Times vs Chunk Overlaps for Different Chunkers (Chunk size=400)')
    plt.grid()
    plt.legend()
    plt.show() """

    """ chunker = "NLTKTextSplitter"
    times = []
    for overlap in chunks_overlaps:
        times.append(results[chunker][f"400-{overlap}"]["time"])
    
    plt.plot(chunks_overlaps, times, label=f"{chunker}")

    plt.xlabel('Chunk Overlaps')
    plt.ylabel('Times [s]')
    plt.title('Times vs Chunk Overlaps for NLTKTextSplitter (Chunk size=400)')
    plt.grid()
    plt.show() """

    """ for batch in chunk_batches:
        times.append(results[chunker][f"{batch}-0.3"]["time"])
    
    plt.plot(chunk_batches, times, label=f"{chunker}")

    plt.xlabel('Chunk Batches')
    plt.ylabel('Times [s]')
    plt.title('Times vs Chunk Batches for NLTKTextSplitter (Overlap=30%)')
    plt.grid()
    plt.show() """

    """ for chunker in chunkers:
        times = []
        for batch in chunk_batches:
            times.append(results[chunker][f"{batch}-0.3"]["n_chunks"])
        plt.plot(chunk_batches, times, label=f"{chunker}")

    plt.xlabel('Chunk Batches')
    plt.ylabel('Number of Chunks')
    plt.title('Number of Chunks vs Chunk Batches for Different Chunkers (Overlap=30%)')
    plt.grid()
    plt.legend()
    plt.show() """

    for chunker in chunkers:
        times = []
        for overlap in chunks_overlaps:
            times.append(results[chunker][f"400-{overlap}"]["n_chunks"])
        plt.plot(chunks_overlaps, times, label=f"{chunker}")
    
    plt.xlabel('Chunk Overlaps')
    plt.ylabel('Number of Chunks')
    plt.title('Number of Chunks vs Chunk Overlaps for Different Chunkers (Chunk size=400)')
    plt.grid()
    plt.legend()
    plt.show()