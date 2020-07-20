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
        plt.plot(training_history[k], label = k + '-train')
        plt.plot(validation_history[k], label = k + '-val')

    plt.xlabel('epochs')
    plt.ylabel('nats/dim')
    plt.title('Performance on Training/Validation Set')
    plt.legend()
