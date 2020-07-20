import torch
import ignite
from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from .loss import kl_loss, recon_loss

def create_basic_trainer(model,optimizer,device,beta=1,**kwargs):
    
    evaluator = kwargs.get('evaluator', None)
    val_loader = kwargs.get('val_loader', None)

    def process_function(engine,x):
        
        model.train()
        optimizer.zero_grad()        
        x = x.to(device)

        x_recon, logstd_noise, mu_z, logstd_z = model(x)
        
        kl = kl_loss(mu_z,logstd_z)
        recon = recon_loss(x, x_recon, logstd_noise, device)
        loss = recon + beta * kl
        
        loss.backward()
        optimizer.step()
        return loss.item(), recon.item(), kl.item()

    trainer = Engine(process_function)

    # Register running averages for main losses
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer,'elbo_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer,'recon_loss')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer,'kl_loss')

    # Attach evaluator if passed
    if evaluator:
        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(engine):
            evaluator.run(val_loader,max_epochs=1)

    return trainer


def create_basic_evaluator(model, device, beta=1, **kwargs):
    
    def evaluate_function(engine,batch):
        model.eval()
        with torch.no_grad():
            x = batch
            x = x.to(device)

            x_recon, logstd_noise, mu_z, logstd_z = model(x)
            kw = { 'logstd_noise': logstd_noise, 'mu_z':mu_z, 'logstd_z': logstd_z}
            return x, x_recon, kw

    evaluator = Engine(evaluate_function)
    
    # Registering metrics
    m1 = Loss(kl_loss, output_transform=lambda x:(x[2]['mu_z'],x[2]['logstd_z']))
    m2 = Loss(recon_loss, output_transform=lambda x:(x[0],x[1],
        {'logstd_noise':x[2]['logstd_noise'],'device':device}))
    m1.attach(evaluator, 'kl_loss')
    m2.attach(evaluator, 'recon_loss')

    m3 = m2 + beta * m1
    m3.attach(evaluator, 'elbo_loss')

    return evaluator
