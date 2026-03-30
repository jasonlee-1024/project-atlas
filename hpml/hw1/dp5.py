import sys
import time
import numpy as np

# main
def main():
    # read input
    n = int(sys.argv[1])
    reps = int(sys.argv[2])
    print(f"Running dot product with N={n} and reps={reps}")

    # initialize arrays
    A = np.ones(n,dtype=np.float32)
    B = np.ones(n,dtype=np.float32)

    total_time = 0.0

    for r in range(reps):
        # get start time 
        start = time.time()

        # compute dot product
        R = np.dot(A,B)

        # get end time
        end = time.time()

        #print time taken for this repetition
        s = end - start
        print(f"Repetition {r+1}: {s:.6f} seconds")

        if r >= reps//2:
            total_time += s
    
    # calculate average time
    avg_time = total_time / (reps//2)
    
    # calculate bandwidth
    bytes_moved = 2 * n * A.itemsize
    bandwidth = bytes_moved / avg_time / 1.0e9

    # calculate flops
    flops = 2 * n / avg_time 

    #print results
    print(f"N: {n}, <T>: {avg_time:.6f} sec, B: {bandwidth:3f} GB/sec, FLOPS: {flops:.3f} FLOP/sec")

        

if __name__ == "__main__":
    main()


