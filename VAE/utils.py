import matplotlib.pyplot as plt

def print_and_collect(engine, evaluator, dataloader, mode, history_dict):
    
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics

    print(
        mode + " Results - Epoch {} - ELBO loss: {:.2f} RECON loss: {:.2f} KL loss: {:.2f}"
        .format(engine.state.epoch, metrics['elbo_loss'], metrics['recon_loss'], 
        metrics['kl_loss']))

    for key in history_dict.keys():
        history_dict[key].append(metrics[key])


def plot_results( training_hist, validation_hist ):
    
    for k in training_hist.keys():
        plt.plot(training_hist[k], label = k + '-train')
        plt.plot(validation_hist[k], label = k + '-val')

    plt.xlabel('epochs')
    plt.ylabel('nats/dim')
    plt.title('Performance on Training/Validation Set')
    plt.legend()

def interpolate_latent(x1,x2,model,k=10):
    """ Interpolates between given samples in the latent space and returns the
    corresponding dataspace samples.
    
    Args:
        x1 (tensor)
        x2 (tensor)
        model (model)
        k (int): no.of interpolation b/w x1 and x2 including x1 and x2
    Returns:
        res (list): list of interpolated points b/w x1 and x2
    """
    
    z1 = model.mapToLatent(x1.unsqueeze(0))
    z2 = model.mapToLatent(x2.unsqueeze(0))

    res = [ model.dec(torch.lerp(z1,z2,w))[0] for w in np.linspace(0,1,k,True) ]
    return res
