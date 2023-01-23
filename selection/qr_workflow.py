import os

import dadrah.selection.discriminator as disc
import dadrah.selection.loss_strategy as lost
import dadrah.util.string_constants_util as stco
import vande.training as train

def train_L1QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR with lambda layer on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    #discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    print("YYYYYY")
    lambda_discriminator = disc.L1QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=1024, epochs=params.epochs,  n_layers=5, n_nodes=60)
    losses_train, losses_valid = lambda_discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return lambda_discriminator




def train_L2QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR with lambda layer on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    #discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    lambda_discriminator = disc.L2QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=1024, epochs=params.epochs,  n_layers=5, n_nodes=60)
    losses_train, losses_valid = lambda_discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return lambda_discriminator


def train_L3QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR with lambda layer on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    #discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    print("YYYYYY")
    lambda_discriminator = disc.L3QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=512, epochs=params.epochs,  n_layers=5, n_nodes=50)
    losses_train, losses_valid = lambda_discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return lambda_discriminator


def train_L4QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR with lambda layer on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    #discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    print("YYYYYY")
#    lambda_discriminator = disc.L4QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=32000, epochs=params.epochs,  n_layers=5, n_nodes=60)
    lambda_discriminator = disc.L4QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=220, epochs=params.epochs,  n_layers=5, n_nodes=60)
    losses_train, losses_valid = lambda_discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return lambda_discriminator


def train_L5QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR with lambda layer on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    #discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    print("YYYYYY")
    lambda_discriminator = disc.L5QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=32000, epochs=params.epochs,  n_layers=5, n_nodes=60)
    losses_train, losses_valid = lambda_discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return lambda_discriminator

def train_L6QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR with lambda layer on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    #discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    print("YYYYYY")
    lambda_discriminator = disc.L6QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=1024, epochs=params.epochs,  n_layers=5, n_nodes=60)
    losses_train, losses_valid = lambda_discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return lambda_discriminator


def train_LBSQR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR with lambda layer on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    #discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=60)
    lambda_discriminator = disc.LBSQRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=150, epochs=params.epochs,  n_layers=5, n_nodes=60)
    losses_train, losses_valid = lambda_discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return lambda_discriminator


def train_QR(quantile, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR on qcd-signal-injected sample and quantile q
    
    print('\ntraining QR for quantile {}'.format(quantile))    
    discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=20000, epochs=params.epochs,  n_layers=8, n_nodes=80)
    losses_train, losses_valid = discriminator.fit(mixed_train_sample, mixed_valid_sample)

    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantile, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')

    return discriminator


def train_VQRv1(quantiles, mixed_train_sample, mixed_valid_sample, params, plot_loss=False):

    # train QR with lambda layer on qcd-signal-injected sample and quantile q
    
    print("###################")
    print(f'\ntraining QR for quantile {quantiles}')
    print("###################")
    #vqr_discriminator = disc.VQRv1Discriminator_KerasAPI(quantiles=quantiles, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=256, epochs=params.epochs,  n_layers=5, n_nodes=30)
    vqr_discriminator = disc.VQRv1Discriminator_KerasAPI(quantiles=quantiles, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=160, epochs=params.epochs,  n_layers=5, n_nodes=10)
    losses_train, losses_valid = vqr_discriminator.fit(mixed_train_sample, mixed_valid_sample)
    
    if plot_loss:
        plot_str = stco.make_qr_model_str(params.run_n, params.sig_sample_id, quantiles, xsec, params.strategy_id)
        train.plot_training_results(losses_train, losses_valid, plot_suffix=plot_str[:-3], fig_dir='fig')
   
    return vqr_discriminator
    

def save_QR(discriminator, params, experiment, quantile, xsec, model_str=None):
    # save the model   
    model_str = model_str or stco.make_qr_model_str(experiment.run_n, quantile, params.sig_sample_id, xsec, params.strategy_id)
    model_path = os.path.join(experiment.model_dir_qr, model_str)
    discriminator.save(model_path)
    print('saving model {} to {}'.format(model_str, experiment.model_dir_qr))
    return model_path


def load_QR(params, experiment, quantile, xsec, date, model_str=None):
    model_str = model_str or stco.make_qr_model_str(experiment.run_n, quantile, params.sig_sample_id, sig_xsec=xsec, strategy_id=params.strategy_id, date=date)
    model_path = os.path.join(experiment.model_dir_qr, model_str)
    discriminator = disc.QRDiscriminator_KerasAPI(quantile=quantile, loss_strategy=lost.loss_strategy_dict[params.strategy_id], batch_sz=300, epochs=params.epochs,  n_layers=5, n_nodes=50)
    discriminator.load(model_path)
    return discriminator


def predict_QR(discriminator, sample, inv_quant):
    print('predicting {}'.format(sample.name))
    selection = discriminator.select(sample)
    sample.add_feature('sel_q{:02}'.format(int(inv_quant*100)), selection)
    return sample


def predict_QR_fold(discriminator, sample, inv_quant,fold):
    print('predicting {}'.format(sample.name))
    selection = discriminator.select(sample)
    sample.add_feature('sel_q{:02}_{}'.format(int(inv_quant*100),fold), selection)
    return sample


def predict_VQR(vdiscriminator, sample, inv_quant, quantileidx, fold):
    print('predicting {}'.format(sample.name))
    selection = vdiscriminator.select(sample, quantileidx)
    sample.add_feature('sel_q{:02}_{}'.format(int(inv_quant*100),fold), selection)
    return sample

