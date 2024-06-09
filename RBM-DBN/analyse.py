
import logging
from utils import *
from model.MNISTDNN import MNISTDNN
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
# # Hyperparameters
v_dim = 784 # 28 * 28
def do_experiment(h_dims, pretrain, data_size, save_pretrain_path='./pretrained/'):
	logging.info(f'do experiment on h_dims={h_dims}, pretrain={pretrain}, data_size={data_size}')
	dnn = MNISTDNN(h_dims=h_dims, pretrain=pretrain, data_dir='./data', data_size=data_size, save_pretrain_path=save_pretrain_path)
	logger = CSVLogger('.')
	trainer = Trainer(logger=logger, max_epochs=200, accelerator='gpu', enable_progress_bar=True)
	trainer.fit(dnn)
	trainer.test(dnn)

def do_experiment_on_layer_depth():
	do_experiment(h_dims=[200,200],pretrain=True, data_size=60000)
	do_experiment(h_dims=[200,200],pretrain=False, data_size=60000)
	do_experiment(h_dims=[200,200,200],pretrain=True, data_size=60000)
	do_experiment(h_dims=[200,200,200],pretrain=False, data_size=60000)
	do_experiment(h_dims=[200,200,200,200],pretrain=True, data_size=60000)
	do_experiment(h_dims=[200,200,200,200],pretrain=False, data_size=60000)
	do_experiment(h_dims=[200,200,200,200,200],pretrain=True, data_size=60000)
	do_experiment(h_dims=[200,200,200,200,200],pretrain=False, data_size=60000)

def do_experiment_on_layer_width(): 
	do_experiment(h_dims=[100,100],pretrain=True, data_size=60000)
	do_experiment(h_dims=[100,100],pretrain=False, data_size=60000)
	do_experiment(h_dims=[300,300],pretrain=True, data_size=60000)
	do_experiment(h_dims=[300,300],pretrain=False, data_size=60000)
	do_experiment(h_dims=[500,500],pretrain=True, data_size=60000)
	do_experiment(h_dims=[500,500],pretrain=False, data_size=60000)
	do_experiment(h_dims=[700,700],pretrain=True, data_size=60000)
	do_experiment(h_dims=[700,700],pretrain=False, data_size=60000)

def do_experiment_on_data_size():
	do_experiment(h_dims=[200,200],pretrain=True, data_size=1000, save_pretrain_path='./pretrained/')
	do_experiment(h_dims=[200,200],pretrain=False, data_size=1000)
	do_experiment(h_dims=[200,200],pretrain=True, data_size=3000, save_pretrain_path='./pretrained/')
	do_experiment(h_dims=[200,200],pretrain=False, data_size=3000)
	do_experiment(h_dims=[200,200],pretrain=True, data_size=7000, save_pretrain_path='./pretrained/')
	do_experiment(h_dims=[200,200],pretrain=False, data_size=7000)
	do_experiment(h_dims=[200,200],pretrain=True, data_size=10000, save_pretrain_path='./pretrained/')
	do_experiment(h_dims=[200,200],pretrain=False, data_size=10000)
	do_experiment(h_dims=[200,200],pretrain=True, data_size=30000, save_pretrain_path='./pretrained/')
	do_experiment(h_dims=[200,200],pretrain=False, data_size=30000)

if __name__ == '__main__':
	logging.basicConfig(filename='analyse.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s')
	do_experiment_on_data_size()
	do_experiment_on_layer_width()
	do_experiment_on_layer_depth()
