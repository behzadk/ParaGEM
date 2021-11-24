import multiprocessing as mp

def a_func(x):
    if x:
        return x
    
    # Function sleeps before returning
    # to trigger timeout error
    else:
        raise mp.context.TimeoutError(x)


if __name__ == "__main__":
    solutions = []

    # Inputs sum to 4
    inputs = [1, 1, 0, 1, 1, 0]

    # with mp.get_context("spawn").Pool(1) as pool:
    #     futures_res = pool.imap(a_func, inputs)
    #     idx = 0
    #     for x in inputs:
    #         try:
    #             res = futures_res.next(timeout=100)
    #             # If successful (no time out), append the result
    #             solutions.append(res)
            
    #         except mp.context.TimeoutError:
    #             print(x, "err")
    #             # Catch time out error
    #             # I want this to also prevent the process from being executed again
    #             # solutions.append(0.0)
    inputs = [1, 1, 0, 0, 1, 1,0, 0,1,0, 1]

    pool = mp.get_context("spawn").Pool(1)
    futures_res = pool.imap(a_func, inputs)
    for x in inputs:
        try:
            res = futures_res.next(timeout=100)
            # If successful (no time out), append the result
            solutions.append(res)
            
        except mp.context.TimeoutError:
            print(x, "err")
            # Catch time out error
            # I want this to also prevent the process from being executed again
            # solutions.append(0.0)

    # Should print 4
    print(len(solutions))
    print(sum(inputs))
    print(sum(solutions))
    
    print("closing")
    pool.close()
    print("joining")
    pool.join()
    print("terminating")
    pool.terminate()

