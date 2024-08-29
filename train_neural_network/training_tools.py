import numpy as np
import matplotlib.pyplot as plt
import math
import time
import torch

def X_from_file(read_folder, dire, graph_gene=None, subset_gene=None):

    # read in all the appropriate files
    c_file = read_folder + "/save_c.txt"
    u_file = read_folder + "/save_u.txt"
    s_file = read_folder + "/save_s.txt"

    c = np.loadtxt(c_file, ndmin=2)
    raw_u = np.loadtxt(u_file, ndmin=2)
    raw_s = np.loadtxt(s_file, ndmin=2)

    alpha_c_file = read_folder+"/alpha_c.txt"
    alpha_file = read_folder+"/alpha.txt"
    beta_file = read_folder+"/beta.txt"
    gamma_file = read_folder+"/gamma.txt"

    epsilon = 1e-5

    alpha_c = np.log(np.loadtxt(alpha_c_file, ndmin=1) + epsilon)
    alpha = np.log(np.loadtxt(alpha_file, ndmin=1) + epsilon)
    beta = np.log(np.loadtxt(beta_file, ndmin=1) + epsilon)
    gamma = np.log(np.loadtxt(gamma_file, ndmin=1) + epsilon)

    model_file = read_folder+"/model_list.txt"

    m = np.loadtxt(model_file, ndmin=1)

    sw_t_file = read_folder+"/switch_times.txt"

    sw_t = np.loadtxt(sw_t_file, ndmin=2)

    t_file = read_folder+"/true_time.txt"

    raw_t = np.loadtxt(t_file, ndmin=2).T

    c0_file = read_folder+"/c0_data.txt"
    u0_file = read_folder+"/u0_data.txt"
    s0_file = read_folder+"/s0_data.txt"

    c0_full = np.loadtxt(c0_file, ndmin=2)
    u0_full = np.log(np.loadtxt(u0_file, ndmin=2) + epsilon)
    s0_full = np.log(np.loadtxt(s0_file, ndmin=2) + epsilon)

    u = np.log(raw_u + epsilon)
    s = np.log(raw_s + epsilon)

    # get the dimensions of the data
    genes = len(alpha_c)

    n = len(c[0])

    print("In this data there are", n, "cells")
    print("and", genes, "genes")

    # X = []

    x_arr = []
    t_arr = []

    gene_len = [0]

    state = np.zeros(n)

    if graph_gene is not None:
        print("Graphing genes", graph_gene)
        plt.scatter(raw_t[:, graph_gene], c[:, graph_gene], s=1)
        plt.xlabel("t")
        plt.ylabel("c")

    last_time = time.time()

    if subset_gene is not None:
        loop_vals = subset_gene
    else:
        loop_vals = range(genes)

    # loop through each gene
    for i in loop_vals:

        if subset_gene is not None and i % 1000 == 0:
            cur_time = time.time()
            print(i, "genes complete with time delta = ", cur_time - last_time)
            last_time = cur_time

        c0 = c0_full[i]
        u0 = u0_full[i]
        s0 = s0_full[i]

        gene_sw_t = sw_t[i]

        gene_t = raw_t[i]

        if graph_gene is not None:
            print("Graphing genes", graph_gene)
            plt.scatter(raw_t[:, graph_gene], c[:, graph_gene], s=1)
            plt.xlabel("t")
            plt.ylabel("c")

        state = np.zeros(n)

        if dire == 0 or dire == 1:
            state1_mask = np.logical_and(gene_t > gene_sw_t[0], gene_t <= gene_sw_t[1])
            state[state1_mask] = 1
        if dire == 0 or dire == 2:
            state2_mask = np.logical_and(gene_t > gene_sw_t[1], gene_t <= gene_sw_t[2])
            state3_mask = raw_t[i] > gene_sw_t[2]
            state[state2_mask] = 2
            state[state3_mask] = 3

        model = m[i]

        max_u = np.max(raw_u[i])
        max_s = np.max(raw_s[i])

        if dire == 0:

            x = np.concatenate((np.array([c[i], u[i],  s[i]]),
                                np.full((n, 17), [alpha_c[i],
                                                  alpha[i],
                                                  beta[i], gamma[i],
                                                c0[1], c0[2], c0[3],
                                                u0[2], u0[3],
                                                s0[2], s0[3],
                                                np.log(max_u + epsilon),
                                                np.log(max_s + epsilon),
                                                gene_sw_t[0],
                                                gene_sw_t[1],
                                                gene_sw_t[2],
                                                model]).T,
                                np.array([state])
                                )).T.astype(np.float32)

            new_state3_mask = np.logical_and(state3_mask,
                                           np.logical_or(raw_u[i] >= max_u * 0.1,
                                          raw_s[i] >= max_s * 0.1))

            no_state_0_mask = np.logical_or(state1_mask,
                                             np.logical_or(state2_mask,
                                                            new_state3_mask))

            x = x[no_state_0_mask]
            t_i = raw_t[i][no_state_0_mask]

        elif dire == 1:

            x = np.concatenate((np.array([c[i], u[i], s[i]]),
                                np.full((n, 12), [alpha_c[i], alpha[i],
                                                  beta[i], gamma[i],
                                                c0[1], c0[2],
                                                u0[1], u0[2],
                                                s0[1], s0[2],
                                                gene_sw_t[0],
                                                model]).T,
                                np.array([state])
                                )).T.astype(np.float32)

            x = x[state1_mask]
            t_i = raw_t[i][state1_mask]

        elif dire == 2:

            if model == 1:

                # this will give us an approximate time value for when
                # u reaches its maximum value
                max_u_t = -((float(1)/np.exp(alpha_c[i]))
                            * np.log((max_u*np.exp(beta[i]))
                                     / (np.exp(alpha[i])*c0[2])))

                x = np.concatenate((np.array([c[i], u[i], s[i]]),
                                    np.full((n, 14), [alpha_c[i],
                                                    alpha[i],
                                                    beta[i],
                                                    gamma[i],
                                                    c0[2],
                                                    c0[3],
                                                    u0[2],
                                                    u0[3],
                                                    s0[2],
                                                    s0[3],
                                                    max_u_t,
                                                    np.log(max_u + epsilon),
                                                    np.log(max_s + epsilon),
                                                    gene_sw_t[2]]).T,
                                    np.array([state])
                                    )).T.astype(np.float32)

            elif model == 2:

                x = np.concatenate((np.array([c[i], u[i], s[i]]),
                                        np.full((n, 12), [alpha_c[i],
                                                        alpha[i],
                                                        beta[i],
                                                        gamma[i],
                                                        # c_0_for_t_guess,
                                                        c0[2], c0[3],
                                                        u0[2],
                                                        u0[3],
                                                        s0[2],
                                                        s0[3],
                                                        np.log(max_u),
                                                        gene_sw_t[2]]).T,
                                        np.array([state])
                                        )).T.astype(np.float32)

            positive_mask = np.logical_or(raw_u[i] >= max_u * 0.1,
                                          raw_s[i] >= max_s * 0.1)

            x = x[positive_mask]
            t_i = raw_t[i][positive_mask]

        x_arr.append(x)
        t_arr.append(t_i)

        if subset_gene is not None:
            gene_len.append(len(t_i))

    start = time.time()
    # print(np.array(X).shape)
    # print(X[-1][-1])

    X = np.concatenate([row for row in x_arr])

    t = np.concatenate([row for row in t_arr])

    if subset_gene is None:
        print(X.shape)
        print("time to ravel:", time.time() - start)

    if subset_gene is not None:
        return X, t, gene_len

    return X, t


def make_batches_random(batches, test_X, test_t):

    # inspired by:
    # https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way

    x_tens = []
    t_tens = []

    np.random.seed(42)

    assert test_X.shape[0] == test_t.shape[0]

    test_N = test_X.shape[0]

    # the "width" of the batch
    w = math.floor(float(test_N) / float(batches))

    X_cols = test_X.shape[1]

    permutation = torch.randperm(test_N)

    for i in range(batches):

        current_set = permutation[i*w:(i+1)*w]

        x_ten = torch.tensor(test_X[current_set], dtype=torch.float, requires_grad=True).reshape(-1, X_cols)
        t_ten = torch.tensor(test_t[current_set], dtype=torch.float, requires_grad=True).reshape(-1, 1)

        x_tens.append(x_ten)
        t_tens.append(t_ten)

    if test_N % w != 0:
        current_set = permutation[(i+1)*w:-1]
        x_ten = torch.tensor(test_X[current_set], dtype=torch.float, requires_grad=True).reshape(-1, X_cols)
        t_ten = torch.tensor(test_t[current_set], dtype=torch.float, requires_grad=True).reshape(-1, 1)

        x_tens.append(x_ten)
        t_tens.append(t_ten)

    return x_tens, t_tens


def split_test_train(X, t, fraction):

    N = X.shape[0]

    test_split = math.floor(N*fraction)

    val_split = N - test_split

    np.random.seed(42)

    full_data = range(N)

    test_choice = np.random.choice(N, size=test_split, replace=False)
    val_choice = np.setdiff1d(full_data, test_choice)

    test_X = X[test_choice]
    test_t = t[test_choice]

    val_X = X[val_choice]
    val_t = t[val_choice]

    print("Test size:", test_X.shape[0])
    print("Validation size:", val_X.shape[0])

    return test_X, test_t, val_X, val_t


def graph_results(t, t_pred, X, title):

    x = np.linspace(np.min(t), np.max(t), 10)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

    ax1.set_title(title)
    ax1.plot(x, x, linewidth=1, label="Perfect fit", color="orange")
    ax1.scatter(t, t_pred, s=1, label="Data")
    ax1.set_xlabel("Multivelo t")
    ax1.set_ylabel("My t")
    ax1.legend()

    ax2.set_title(title)
    ax2.scatter(t, X[:,0], label="Correct", s=2, color="Orange")
    ax2.scatter(t_pred, X[:,0], s=1, label="ML Data", color="Blue")
    ax2.legend()
    ax2.set_xlabel("Time")
    ax2.set_ylabel("C")

    ax3.set_title(title)
    ax3.scatter(t, X[:, 1], label="Correct", s=2, color="Orange")
    ax3.scatter(t_pred, X[:, 1], s=1, label="ML Data", color="Blue")
    ax3.legend()
    ax3.set_xlabel("Time")
    ax3.set_ylabel("U")

    ax4.set_title(title)
    ax4.scatter(t, X[:, 2], label="Correct", s=2, color="Orange")
    ax4.scatter(t_pred, X[:, 2], s=1, label="ML Data", color="Blue")
    ax4.legend()
    ax4.set_xlabel("Time")
    ax4.set_ylabel("S")

    plt.show()