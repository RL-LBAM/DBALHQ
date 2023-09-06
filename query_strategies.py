import torch
import numpy as np
from scipy import stats
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from scipy.spatial.distance import pdist, squareform


def predictions_from_pool(
    model, X_pool: np.ndarray, T: int = 100,  pool_size: int = 2000
):
    # Randomly select points from the pool set as the actual pool. Conduct serveral forward propagations on the pool.

    random_subset = np.random.choice(range(len(X_pool)), size=pool_size, replace=False)
    with torch.no_grad():
        outputs = np.stack(
            [
                torch.softmax(
                    model.estimator.forward(X_pool[random_subset], training=True),
                    dim=-1,
                )
                .cpu()
                .numpy()
                for _ in range(T)
            ]
        )



    # Reset batch size for training to get embeddings
    model.estimator.batch_size=int(np.max(np.array([len(model.X_training), pool_size])))

    # Embeddings of annotated points
    model.estimator.forward(model.X_training,training=False)
    label_eb=model.estimator.module.e.cpu().numpy()

    # Embeddings of unannotated points
    model.estimator.forward(X_pool[random_subset],training=False)
    unlabel_eb=model.estimator.module.e.cpu().numpy()
    
    # Stack the two embedding matrices, normalize each feature and re-split them
    all_eb=np.vstack((unlabel_eb,label_eb))
    all_norm=(all_eb-all_eb.mean(axis=0))/all_eb.std(axis=0)
    unlabel_norm, label_norm=np.vsplit(all_norm, [pool_size])

    return outputs, random_subset, label_norm, unlabel_norm



def combine_metric(model, X_pool: np.ndarray, uncertainty,diversity, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    # Combine metrics using the weighted mean methods

    outputs, random_subset, label_norm, unlabel_norm = predictions_from_pool(model, X_pool, T, pool_size)

    uncertainty_values=uncertainty(outputs).reshape((-1,))
    diversity_values=diversity(model, label_norm, unlabel_norm).reshape((-1,))
    
    # Whether to use time-decayed weight
    if model.args.time_decay==False:
        beta=model.beta
    else:
        beta = 1-np.exp(-model.beta*model.time)

    # Which mean to use
    if model.args.ari_geo == 1:
        final_score=np.power(uncertainty_values, beta)*np.power(diversity_values, 1-beta)
    else:
        final_score=beta*uncertainty_values+(1-beta)*diversity_values

    idx = (-final_score).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def two_stage_query(model, X_pool: np.ndarray, uncertainty,diversity, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    # Combine metrics using the two stage methods
    outputs, random_subset, label_norm, unlabel_norm = predictions_from_pool(model, X_pool, T, pool_size)
    
    uncertainty_values=uncertainty(outputs).reshape((-1,))
    diversity_values=diversity(model, label_norm, unlabel_norm).reshape((-1,))

    # Check metric order
    if model.args.priority == 0:
        idx1 = -(uncertainty_values).argsort()[:(n_query*model.ratio)]
        candidates=diversity_values[idx1]
        idx2= -(candidates).argsort()[:n_query]
        idx3=idx1[idx2]
        query_idx=random_subset[idx3]
        return query_idx, X_pool[query_idx]
    else:
        idx1 = -(diversity_values).argsort()[:(n_query*model.ratio)]
        candidates=uncertainty_values[idx1]
        idx2= -(candidates).argsort()[:n_query]
        idx3=idx1[idx2]
        query_idx=random_subset[idx3]
        return query_idx, X_pool[query_idx]


# Each possible hybrid query strategy. The name is the abbervation of methods considered. For example, ed = entropy + disscore. They will check the runmode 
# to decide combination methods

def ed(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, entropy, disscore,n_query, T, pool_size)
    else:
        return combine_metric(model, X_pool, entropy, disscore,n_query, T, pool_size)

def ep(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, entropy, posterior,n_query, T, pool_size)
    else:
        return combine_metric(model, X_pool, entropy, posterior,n_query, T, pool_size)

def em(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, entropy, minidis,n_query, T,  pool_size)
    else:
        return combine_metric(model, X_pool, entropy, minidis, n_query, T, pool_size)

def bd(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100,  pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, bald, disscore,n_query, T,  pool_size)
    else:
        return combine_metric(model, X_pool, bald, disscore, n_query, T, pool_size)

def bp(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, bald, posterior,n_query, T, pool_size)
    else:
        return combine_metric(model, X_pool, bald, posterior,n_query, T, pool_size)

def bm(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, bald, minidis,n_query, T, pool_size)
    else:
        return combine_metric(model, X_pool, bald, minidis,n_query, T, pool_size)

def vd(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, var_ratios, disscore,n_query, T, pool_size)
    else:
        return combine_metric(model, X_pool, var_ratios, disscore,n_query, T, pool_size)

def vp(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, var_ratios, posterior,n_query, T, pool_size)
    else:
        return combine_metric(model, X_pool, var_ratios, posterior,n_query, T, pool_size)

def vm(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    if model.args.runmode!=0:
        return two_stage_query(model, X_pool, var_ratios, minidis,n_query, T, pool_size)
    else:
        return combine_metric(model, X_pool, var_ratios, minidis, n_query, T, pool_size)

def uniform(model, X_pool: np.ndarray, n_query: int = 10,T: int = 100, pool_size: int = 2000):
    # The uniform baseline

    outputs, random_subset, label_norm, unlabel_norm = predictions_from_pool(model, X_pool, T, pool_size)
    query_idx = np.random.choice(random_subset, size=n_query, replace=False)
    return query_idx, X_pool[query_idx]



def shannon_entropy_function(outputs,E_H=False):
    # Calculate entropy or BALD of each point

    pc = outputs.mean(axis=0)
    # Upperbound of the metric values
    upperbound=np.log(10)

    # To avoid numeric underflow, add 1e-10
    H = (-pc * np.log(pc + 1e-10)).sum(
        axis=-1
    ) 
    if E_H:
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E

    return H/upperbound


def entropy(outputs):
    # Uncertainty measured by entropy
    
    H = shannon_entropy_function(outputs)
    return H


def bald(outputs):
    # Uncertainty measured by BALD
    
    upperbound=np.log(10)

    H, E_H = shannon_entropy_function(outputs, E_H=True)
    
    return (H-E_H)/upperbound


def var_ratios(outputs):
    # Uncertainty measured by variation ratios
    
    upperbound=0.9
    preds = np.argmax(outputs, axis=2)
    _, count = stats.mode(preds, axis=0, keepdims=True)
    ratio = (1 - count / preds.shape[1]).reshape((-1,))
    ratio=ratio/upperbound
    return ratio

def disscore(model, label_norm, unlabel_norm):
    # Diversity measured by discriminator score

    # Upsample annotated points to balance labels
    label_len=label_norm.shape[0]
    unlabel_len=unlabel_norm.shape[0]
    ratio=int(np.round(unlabel_len/label_len))
    label_norm=np.tile(label_norm,(ratio,1))

    # Create labels
    label_target=np.ones(label_norm.shape[0]).reshape(-1,1)
    unlabel_target=np.zeros(unlabel_norm.shape[0]).reshape(-1,1)
    all_features=np.vstack((label_norm,unlabel_norm))
    all_targets=np.vstack((label_target,unlabel_target)).reshape(-1).astype(np.int64)
    
    # Train the discriminator and get diversity scores
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    discri1 = discri().to(device)
    discriminator = NeuralNetClassifier(
        module=discri1,
        lr=model.args.lr,
        batch_size=model.args.batch_size,
        max_epochs=model.args.epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device
    )

    discriminator.fit(all_features,all_targets)
    scores=discriminator.predict_proba(all_features)[-unlabel_len:,0]
    
    # Report numeric errors
    if ((scores>=0).all())==False:
        print('error: negative diversity score\nminimum value:')
        print(np.min(scores))
        scores[scores < 0]=0
    
    return(scores)


def posterior(model, label_norm, unlabel_norm):
    # Diversity measured by posterior variance
    
    # Calculate covariance matrix of annotated points
    pairwise_dists = squareform(pdist(label_norm, 'euclidean'))
    cov_matrix=np.exp((-pairwise_dists**2)/20)

    # Calculate kernal matrix. Suppose we have M annotated points and N unannotated points. 
    # The final matrix is N*M where value in (i, j) is k(unlabel_i, label_j)
    unlabel_new=np.expand_dims(unlabel_norm,axis=1)
    kernel_vec_matrix=np.exp(-((label_norm-unlabel_new)**2).sum(2)/20)

    # Avoid numeric errors in matrix inversion
    try:
        va_decrease=(kernel_vec_matrix*(np.linalg.solve(cov_matrix,kernel_vec_matrix.T)).T).sum(1)
    except:
        va_decrease=0

    scores=1-va_decrease
    
    if ((scores>=0).all())==False:
        print('error: negative diversity score\nminimum value:')
        print(np.min(scores))
        scores[scores < 0]=0

    return(scores)

def minidis(model, label_norm, unlabel_norm):
    # Diversity measured by minimum distance 

    unlabel_new=np.expand_dims(unlabel_norm,axis=1)
    scores=(((label_norm-unlabel_new)**2).sum(2)).min(1)
    scores=np.sqrt(scores)
    upperbound=np.max(scores)

    if ((scores>=0).all())==False:
        print('error: negative diversity score\nminimum value:')
        print(np.min(scores))
        scores[scores < 0]=0

    return(scores/upperbound)


class discri(nn.Module):
    # The discriminator to judge whether a point is annotated or not. One hidden layer.

    def __init__(
        self,
        dense1: int =10,
        dense2: int =10,
        target: int =2
    ):
        super(discri, self).__init__()
        self.fc1=nn.Linear(dense1,dense2)
        self.fc2=nn.Linear(dense2,target)
        self.dropout1 = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        out = self.fc2(x)
        
        return out





