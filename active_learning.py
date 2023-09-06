import torch
import time
import numpy as np
from modAL.models import ActiveLearner
from scipy import stats
from query_strategies import ed, ep, em, bd, bp, bm, vd, vp, vm, uniform


def active_learning_procedure(
    query_strategy,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_init: np.ndarray,
    y_init: np.ndarray,
    estimator,
    args,
    beta,
    ratio,
    totest
):
    # Conduct one experiment for one query strategy
    
    T=args.dropout_iter
    n_query=args.query_number
    query_times=args.query_times
    pool_size=args.pool_size

    learner = ActiveLearner(
        estimator=estimator,
        X_training=X_init,
        y_training=y_init,
        query_strategy=query_strategy
    )

    # Pass arguments to the query strategy
    learner.args=args
    learner.beta=beta
    learner.ratio=ratio


    # Record performance on the validation set to tune hyperparameters
    if totest == False:

        for index in range(query_times):
            # Record query times for time-decayed weighted mean combination methods
            learner.time=index+1
            # Query points from the pool
            query_idx, query_instance = learner.query(
                X_pool, n_query=n_query, T=T, pool_size=pool_size
            )
            # Reset batch size for training and retrain the mdoel
            learner.estimator.batch_size=args.batch_size
            learner.teach(X_pool[query_idx], y_pool[query_idx])
            
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            
            if (index + 1) % 5 == 0:
                print(f"query {index+1}/{query_times} completed")
        
        # Calculate prediction accuracy in a deteriministic way to reduce computational cost. 
        # Use cal_acc function below if wanting a fully Bayesian version
        model_accuracy=learner.score(X_val,y_val)
        
        print(f"********** Validation Accuracy per experiment: {model_accuracy} **********")
        
        # Accuracy cureves are not needed for hyperparameter tuning
        return 1, model_accuracy

    else:
        # Record performance on the test set to compare query strategies
        perf_hist = [learner.score(X_test,y_test)]
        
        for index in range(query_times):
            learner.time=index+1
            query_idx, query_instance = learner.query(
                X_pool, n_query=n_query, T=T, pool_size=pool_size
            )

            learner.estimator.batch_size=args.batch_size

            learner.teach(X_pool[query_idx], y_pool[query_idx])
            
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            
            if (index + 1) % 5 == 0:
                print(f"query {index+1}/{query_times} completed")
            
            model_accuracy = learner.score(X_test,y_test)
            perf_hist.append(model_accuracy)
            
    
        print(f"********** Test Accuracy per experiment: {model_accuracy} **********")
        return perf_hist, model_accuracy



def select_query(query: tuple = (0,0)):
    # Choose the query strategy to be tested

    query_dict = {
        (0,0): [ed],
        (0,1): [ep],
        (0,2): [em],
        (1,0): [bd],
        (1,1): [bp],
        (1,2): [bm],
        (2,0): [vd],
        (2,1): [vp],
        (2,2): [vm],
        (10,10):[uniform],
        (100,100):[ed, ep, em, bd, bp, bm, vd, vp, vm, uniform]
    }
    return query_dict[query]

def cal_acc(estimator, x, y, T: int = 100):
    # Calculate prediction accuracy in a fully Bayesian way

    with torch.no_grad():
        outputs = np.stack(
            [
                torch.softmax(
                    estimator.forward(x, training=True),
                    dim=-1,
                )
                .cpu()
                .numpy()
                for _ in range(T)
            ]
        )

    # Make predictions using samples
    preds = np.argmax(outputs, axis=2)
    preds1, _ = stats.mode(preds, axis=0,keepdims=True)

    return((y==preds1).mean())