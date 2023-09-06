import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from scipy.special import logit

from load_data import LoadData, setup_seed
from cnn_model import ConvNNMNIST, ConvNNCIFAR10
from active_learning import select_query, active_learning_procedure


# The codes are adapted from https://github.com/lunayht/DBALwithImgData


def load_CNN_model(args, device):
    # Load the model for different datasets
    
    if args.dataset == 0:
        ConvNN=ConvNNMNIST
    elif args.dataset==1:
        ConvNN=ConvNNCIFAR10

    model = ConvNN().to(device)
    cnn_classifier = NeuralNetClassifier(
        module=model,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device
    )
    return cnn_classifier


def save_as_npy(data: np.ndarray, folder: str, name: str):
    # Save results as npy file
    
    file_name = os.path.join(folder, name + ".npy")
    np.save(file_name, data)
    print(f"Saved: {file_name}")


def plot_results(data: dict):
    # Plot accuracy curves with standard errors

    for key in data.keys():
        experiment,query_times=data[key].shape
        se=np.std(data[key],axis=0)/np.sqrt(experiment)
        mean=np.mean(data[key],axis=0)
        plt.plot(np.arange(query_times),mean,label=key)
        plt.fill_between(np.arange(query_times), mean - se, mean + se,alpha=0.1)

    plt.legend()
    plt.grid(False)
    plt.xlabel('query times')
    plt.ylabel('test accuracy')
    plt.show()


def print_elapsed_time(start_time: float, exp: int, query_stra: str):
    # Print elapsed time for each experiment

    elp = time.time() - start_time
    print(
        f"********** Experiment {exp} ({query_stra}): {int(elp//3600)}:{int(elp%3600//60)}:{int(elp%60)} **********"
    )

def experiment_iter(args, device, datasets: dict, query_stra, beta: float,ratio: int,seeds: int,totest: bool):
    # Conduct experiments for one query strategy
    
    avg_hist=[]
    scores = []
    
    for e in range(args.experiments):
        # Set the random seed before each experiment to ensure the results are reproducible and comparable

        setup_seed(int(seeds[e]))
        start_time = time.time()
        estimator = load_CNN_model(args, device)

        print(
            f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
        )
        
        training_hist, val_score = active_learning_procedure(
            query_strategy=query_stra,
            X_val=datasets["X_val"],
            y_val=datasets["y_val"],
            X_test=datasets["X_test"],
            y_test=datasets["y_test"],
            X_pool=datasets["X_pool"],
            y_pool=datasets["y_pool"],
            X_init=datasets["X_init"],
            y_init=datasets["y_init"],
            estimator=estimator,
            args=args,
            beta=beta,
            ratio=ratio,
            totest=totest
        )
        avg_hist.append(training_hist)
        scores.append(val_score)
        print_elapsed_time(start_time, e + 1, query_stra)

    avg_score=sum(scores) / len(scores)
    
    # return final valiation accuracy when conducting hyperparameter tuning, return final test accuracy and accuracy curves otherwise
    if totest == False:
        print('current accuracy:' + str(avg_score))
        return avg_score
    else:
        print('final accuracy:'+ str(avg_score))
        avg_hist = np.array(avg_hist)
        return avg_hist, avg_score


def train_active_learning(args, device, datasets: dict):
    # Conduct experiments and save results for different query strategies

    query_strategies = select_query((args.uncertainty,args.diversity))

    # Set random seeds for each experiment
    seeds=np.random.choice(range(10000),size=args.experiments,replace=False)
    
    # Dict to record results
    results = dict()
    result_para = dict()

    # Set weighted mean combination methods

    if args.runmode == 0:
        print("This is a weighted mean combination method")
        if args.time_decay == True:
            print("This is time decay version")
        else:
            print("This is constant weight version")

        if args.ari_geo == 1:
            print("This is weighted geometric mean")
        else:
            print("This is weighted arithmetic mean")

        # If the hyperparameter is not set, conduct hyperparameter tuning. Get the baseline results when setting the hyperparameter to 1/0

        if args.beta==100:
            print("No alpha/beta specified, conduct hyperparameter tuning")
            
            for query_stra in query_strategies:
                query_stra_name0 = str(query_stra).split(" ")[1]
                query_stra_name = str(query_stra).split(" ")[1]

                if args.time_decay==True:
                    query_stra_name+="time_decay"

                if args.ari_geo == 1:
                    query_stra_name+="geo"
                else:
                    query_stra_name+="ari"

                # Uniform is the final one when conducting experiments for all query strategies

                if str(query_stra).split(" ")[1] == "uniform":
                    avg_hist, avg_test =experiment_iter(args=args, device=device, datasets=datasets, query_stra=query_stra, beta=args.beta,
                        ratio=args.candidate_ratio,seeds=seeds,totest=True)
                    results[query_stra_name0] = avg_hist
                    result_para[query_stra_name0]=np.array([avg_test])
                    break 

                # Hyperparameter tuning using grid search

                beta_test=np.linspace(0.1, 0.9, 9)
                beta_test_value=[]

                for b in beta_test:
                    print('current alpha/beta: '+str(b))
                    avg_val=experiment_iter(args=args, device=device, datasets=datasets, query_stra=query_stra, beta=b,
                        ratio=args.candidate_ratio,seeds=seeds, totest=False)
                    beta_test_value.append(avg_val)

                
                # Find the best hyperparameter value and calculate the test accuracy
                idx=np.argsort(-np.array(beta_test_value))[0]
                beta_final=beta_test[idx]
                print('final alpha/beta:'+str(beta_final))

                avg_hist, avg_test =experiment_iter(args=args, device=device, datasets=datasets, query_stra=query_stra,beta=beta_final,
                    ratio=args.candidate_ratio,seeds=seeds,totest=True)

                """
                Results saves the accuracy curves in different experiments. Result_para saves the best hyperparameter value and final test accuracy
                """
                results[query_stra_name0] = avg_hist
                result_para[query_stra_name0]=np.array([beta_final, avg_test])
            
            if args.dataset==0:
                query_stra_name+='MNIST'
            else:
                query_stra_name+='CIFAR-10'

            save_as_npy(data=results, folder=args.result_dir, name=query_stra_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=query_stra_name+"para")

        else:
            print("alpha/beta is specified, directly run the model")
            for query_stra in query_strategies:
                query_stra_name0 = str(query_stra).split(" ")[1]
    
                if args.time_decay==False:
                    query_stra_name = str(query_stra).split(" ")[1] + "-betagiven"+str(args.beta)
                else:
                    query_stra_name = str(query_stra).split(" ")[1] + "-betagiven"+str(args.beta)+"-decay"

                if args.ari_geo == 1:
                    query_stra_name+="geo"
                else:
                    query_stra_name+="ari"

                # Directly record test accuracy when the hyperparameter value is given

                avg_hist, avg_test=experiment_iter(args=args, device=device, datasets=datasets, query_stra=query_stra, beta=args.beta,
                    ratio=args.candidate_ratio,seeds=seeds,totest=True)

                results[query_stra_name0] = avg_hist
                result_para[query_stra_name0]=np.array([args.beta, avg_test])

            if args.dataset==0:
                query_stra_name+='MNIST'
            else:
                query_stra_name+='CIFAR-10'

            save_as_npy(data=results, folder=args.result_dir, name=query_stra_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=query_stra_name+"para")

    else:

        print("This is a two-stage combination method")

        # Set two-stage combination methods

        if args.priority == 0:
            print("This is uncertainty first search")
        else:
            print("This is diversity first search")

        if args.candidate_ratio == 100:
            print("No ratio specified, conduct hyperparameter tuning")

            for query_stra in query_strategies:
                query_stra_name0 = str(query_stra).split(" ")[1]
                query_stra_name = str(query_stra).split(" ")[1]

                if args.priority == 0:
                    query_stra_name+="uncertainty"
                else:
                    query_stra_name+="diverisity"

                if str(query_stra).split(" ")[1] == "uniform":
                    avg_hist, avg_test =experiment_iter(args=args, device=device, datasets=datasets, query_stra=query_stra,beta=args.beta,
                        ratio=args.candidate_ratio,seeds=seeds,totest=True)
                    results[query_stra_name0] = avg_hist
                    result_para[query_stra_name0]=np.array([avg_test])
                    break

                # Hyperparameter tuning by search some plausible values
                ratio_test=[2, 3, 4, 5, 6]
                ratio_test_value=[]

                for ratio in ratio_test:
                    print('current ratio'+str(ratio))

                    avg_val=experiment_iter(args=args, device=device, datasets=datasets, query_stra=query_stra,beta=args.beta,
                        ratio=ratio,seeds=seeds, totest=False)

                    ratio_test_value.append(avg_val)

                idx=np.argsort(-np.array(ratio_test_value))[0]
                ratio_final=ratio_test[idx]
                print('final ratio:'+str(ratio_final))

                avg_hist, avg_test=experiment_iter(args=args, device=device, datasets=datasets, query_stra=query_stra,beta=args.beta,
                    ratio=ratio_final,seeds=seeds,totest=True)

                results[query_stra_name0] = avg_hist
                result_para[query_stra_name0]=np.array([ratio_final,avg_test])

            if args.dataset==0:
                query_stra_name+='MNIST'
            else:
                query_stra_name+='CIFAR-10'

            save_as_npy(data=results, folder=args.result_dir, name=query_stra_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=query_stra_name+"para")

        else:
            print("ratio is specified, directly run the model")

            for query_stra in query_strategies:
                query_stra_name0 = str(query_stra).split(" ")[1]
                query_stra_name = str(query_stra).split(" ")[1] +"ratio_given"+str(args.candidate_ratio)

                if args.priority == 0:
                    query_stra_name+="uncertainty"
                else:
                    query_stra_name+="diverisity"

                avg_hist, avg_test=experiment_iter(args=args, device=device, datasets=datasets, query_stra=query_stra,beta=args.beta,
                    ratio=args.candidate_ratio,seeds=seeds,totest=True)

                results[query_stra_name0] = avg_hist
                result_para[query_stra_name0]=np.array([args.candidate_ratio, avg_test])

            if args.dataset==0:
                query_stra_name+='MNIST'
            else:
                query_stra_name+='CIFAR-10'

            save_as_npy(data=results, folder=args.result_dir, name=query_stra_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=query_stra_name+"para")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size in training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=369, 
        help="random seed"
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=5,
        help="number of experiments for each query strategy",
    )
    parser.add_argument(
        "--dropout_iter",
        type=int,
        default=100,
        help="number of forward propagation when making predictions",
    )

    parser.add_argument(
        "--query_times",
        type=int,
        default=20,
        help="times of query",
    )

    parser.add_argument(
        "--query_number",
        type=int,
        default=10,
        help="data batch size per query",
    )
    
    parser.add_argument(
        "--pool_size",
        type=int,
        default=2000,
        help="actual pool size for query",
    )

    parser.add_argument(
        "--uncertainty",
        type=int,
        default=0,
        help="uncertainty metric: 0 = entropy, 1 = bald, 2 = var_ratios, 10 = uniform, 100 = all ",
    )

    parser.add_argument(
        "--diversity",
        type=int,
        default=0,
        help="diverisity metric: 0 = discriminator score, 1 = posterior variance, 2 = minimum distance, 10 = uniform, 100 = all ",
    )

    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        help="validation set size",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default="result_npy",
        help="save npy file in this folder",
    )
    
    parser.add_argument(
        "--runmode",
        type=int,
        default=0,
        help="whether to use weighted mean or two stage combination methods, 0 = weighted, 1 = two stage")

    parser.add_argument(
        "--time_decay",
        type=bool,
        default=False,
        help="whether to decay the weight of diversity with time",
    )

    parser.add_argument(
        "--ari_geo",
        type=int,
        default=0,
        help="specify the type of weighted mean, 0 = arithmetic, 1 = geometric")

    parser.add_argument(
        "--beta",
        type=float,
        default=100,
        help="specify alpha/beta for weighted mean, conduct hyperparamer tuning if not specified")
    
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="set priority of metrics for two stage: 0 = uncertainty first, 1 = diversity first")


    parser.add_argument(
        "--candidate_ratio",
        type=int,
        default=100,
        help="specify gamma for two stage, conduct hyperparamer tuning if not specified")

    parser.add_argument(
        "--dataset",
        type=int,
        default=0,
        help="Dataset, 0 = MNIST, 1 = CIFAR-10")

    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = dict()

    if args.dataset == 0:
        DataLoader=LoadData('MNIST',args.val_size)
    elif args.dataset == 1:
        DataLoader=LoadData('CIFAR10',args.val_size)

    (
        datasets["X_init"],
        datasets["y_init"],
        datasets["X_val"],
        datasets["y_val"],
        datasets["X_pool"],
        datasets["y_pool"],
        datasets["X_test"],
        datasets["y_test"],
    ) = DataLoader.load_all()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    results = train_active_learning(args, device, datasets)

if __name__ == "__main__":
    main()
