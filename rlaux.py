
import gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gym import spaces
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from tqdm import tqdm
import copy
import random
import sys
from stable_baselines3 import PPO
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from create_dataset import CIFAR100
from torch.utils.data import Dataset

# Load Datasets

class LabelTransformDataset(Dataset):
    def __init__(self, dataset, label_transform=None):
        self.dataset = dataset  # Store the original dataset
        self.label_transform = label_transform  # Transform for labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the data and label from the original dataset
        sample, label = self.dataset[idx]
        # Apply the label transform if provided

        if self.label_transform:
            label = self.label_transform(label)
        return sample, label


# load CIFAR100 dataset
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])
trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])


# load CIFAR-100 dataset with batch-size 100
# set keyword download=True at the first time to download the dataset
cifar100_train_set = CIFAR100(root='dataset', train=True, transform=trans_train, download=True)
cifar100_test_set = CIFAR100(root='dataset', train=False, transform=trans_test, download=True)

cifar100_train_set = LabelTransformDataset(cifar100_train_set, label_transform=lambda x: x[ 2].astype(np.int64) )
cifar100_test_set = LabelTransformDataset(cifar100_test_set, label_transform=lambda x: x[ 2].astype(np.int64) )
batch_size = 100
kwargs = {'num_workers': 1, 'pin_memory': True}
cifar100_train_loader = torch.utils.data.DataLoader(
    dataset=cifar100_train_set,
    batch_size=batch_size,
    shuffle=True)

cifar100_test_loader = torch.utils.data.DataLoader(
    dataset=cifar100_test_set,
    batch_size=batch_size,
    shuffle=True)

# Define VGG net
class VGG16(nn.Module):
    def __init__(self, psi):
        super(VGG16, self).__init__()
        """
            multi-task network:
            takes the input and predicts primary and auxiliary labels (same network structure as in human)
        """
        filter = [64, 128, 256, 512, 512]

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # primary task prediction
        self.classifier1 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], len(psi)),
            #nn.Softmax(dim=1)
        )

        # auxiliary task prediction
        self.classifier2 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(psi))),
            #nn.Softmax(dim=1)
        )

        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block

    def forward(self, x):
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        t1_pred = self.classifier1(g_block5.view(g_block5.size(0), -1))
        t2_pred = self.classifier2(g_block5.view(g_block5.size(0), -1))
        return t1_pred, t2_pred

    def model_fit(self, x_pred, x_output, pri=True, num_output=3):
        if not pri:
            # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
            x_output_onehot = x_output
        else:
            # convert a single label into a one-hot vector
            x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
            x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply focal loss
        loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)

    def model_entropy(self, x_pred1):
        # compute entropy loss
        x_pred1 = torch.mean(x_pred1, dim=0)
        loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
       
        return torch.sum(loss1)


# Define Label Generation Network
class LabelNet(nn.Module):
    def __init__(self, features_dim: int, pretrained: bool = False):
        super(LabelNet, self).__init__()
        
        # Load the ResNet-50 model
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace the final fully connected layer to output features_dim
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, features_dim)
        
    def forward(self, x):
        return self.resnet(x), 0
    
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim*64)
        
        #self.net= LabelNet(features_dim=features_dim)
        self.net = VGG16([1]*128)
    def forward(self, observations):
        observations=observations['image']
        outs=[]
        for idx in range(observations.shape[0]):
            obs=observations[idx]
            x= self.net(obs)[0]
            outs.append(x.flatten())
        outs= torch.stack(outs,dim=0)
        return outs
    
class CombinedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CombinedFeatureExtractor, self).__init__(observation_space, features_dim*64)
        
        self.net= LabelNet(features_dim=features_dim)  
        self.embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=50)
        self.combine = nn.Sequential(nn.Linear(178, 128), nn.ReLU(), nn.Linear(128, 128))
    def forward(self, observations):
        labels= observations['array'].type(torch.LongTensor).to(device)  
        observations=observations['image']
        outs=[]
        for idx in range(observations.shape[0]):
            obs=observations[idx]
            x= self.net(obs)[0]
            label_embeddings=self.embedding_layer(labels[idx])
            x=torch.cat([x,label_embeddings],dim=1)
            x=self.combine(x)
            outs.append(x.flatten())
        outs= torch.stack(outs,dim=0)

        return outs
    
class ActionNet(nn.Module):
    def __init__(self):
        super(ActionNet, self).__init__()
        self.fc = nn.Linear(128, aux_dim)
 
    def forward(self, x):
        x=x.reshape(-1,BATCH_SIZE,128)
        return self.fc(x).reshape(-1,BATCH_SIZE*aux_dim)

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc = nn.Linear(128, 1)
 
    def forward(self, x):
        x=x.reshape(-1,BATCH_SIZE,128)
        x=self.fc(x)
        x=x.squeeze(-1).mean(dim=-1)
        return x


class SeededSubsetRandomSampler(SubsetRandomSampler):
    def __init__(self, indices, seed=None):
        self.seed = seed
        super().__init__(indices)

    def __iter__(self):
        if self.seed is not None:
            # Use local random state to shuffle indices based on the given seed
            np_random_state = np.random.RandomState(self.seed)
            indices = list(self.indices)
            np_random_state.shuffle(indices)
            return iter(indices)
        else:
            # Default behavior if no seed is provided
            return super().__iter__()
        
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            class_output, _ = model(inputs)
            loss = criterion(class_output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(class_output, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy

# define RL environment

class AuxTaskEnv(gym.Env):
    def __init__(self, train_dataset, device,model,criterion, optimizer_func, scheduler_func,batch_size=64,aux_dim=50,verbose=False,aux_weight = 1):
        super(AuxTaskEnv, self).__init__()
        
        # save parameters
        self.batch_size=batch_size
        self.train_dataset=train_dataset 
        self.cannonical_model=copy.deepcopy(model)
        self.criterion = criterion
        self.verbose=verbose
        self.aux_weight=aux_weight
        # optimizer_func is a function that takes in a model and gives an optimizer
        self.optimizer_func=optimizer_func
        self.scheduler_func=scheduler_func

        # optimizer and its reload state
        self.optimizer_reload_state=None
        self.optimizer = None
        self.scheduler_reload_state = None
        self.scheduler = None
        
        # Initialize environment variables
        self.device = device
        
        # initialize dataloader with seed
        self.randomize_seed()
        self.reset_data_loader(self.seed)
        self.current_batch = None
    
        # create endless sampler for reward sampling 
        sampler = RandomSampler(train_dataset, replacement=True,num_samples=sys.maxsize )
        self.reward_sampler = iter( DataLoader(train_dataset, batch_size=256, sampler=sampler))

        
        # Define action and observation space
        image_obs = spaces.Box(low=0, high=1, shape=(batch_size, 3, 32, 32), dtype=np.float32)
        label_obs = spaces.Box(low=0, high=100, shape=(100,), dtype=np.float32)

        #self.observation_space = spaces.Box(low=0, high=1, shape=(batch_size, 3, 32, 32), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": image_obs,
            "array": label_obs,
        })
        
        print(self.observation_space)
        self.action_space = spaces.Box(low=-1, high=1, shape=(batch_size, aux_dim), dtype=np.float32)

        # Model for the main classification task (reset this each episode)
        self.model=copy.deepcopy(self.cannonical_model).to(self.device)

        
        # step counter and return counter 
        self.count=0
        self.return_ = 0
        
    def randomize_seed(self):
        self.seed = random.randint(0, 2**32 - 1)

    def reset_data_loader(self,seed):
        # Define the indices for the dataset
        indices = list(range(len(self.train_dataset)))
        sampler = SeededSubsetRandomSampler(indices, seed=seed)

        self.train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)
 
    def get_stats(self):
        print("return:",self.return_)
        print("len",self.count)

    def update(self):
        self.scheduler.step()
        self.cannonical_model=copy.deepcopy(self.model)
        self.optimizer_reload_state=copy.deepcopy( self.optimizer.state_dict())
        self.scheduler_reload_state=copy.deepcopy( self.scheduler.state_dict())
        self.randomize_seed()      

    def evaluate(self,test_loader):
        # Evaluate the network on train and test sets
        if self.model is not None:
          train_accuracy = evaluate(self.model, self.train_loader,self.criterion, self.device)
          test_accuracy = evaluate(self.model, test_loader,self.criterion, self.device)

          print("Train Accuracy",train_accuracy)
          print("Test Accuracy",test_accuracy)

    def reset(self):

        # reset counters
        self.return_ = 0
        self.count = 0

        # Initialize a new main task model from scratch at the start of each episode
        # restore state to cannonical model and optimzizer 
        self.model=copy.deepcopy(self.cannonical_model).to(self.device)
        self.optimizer = self.optimizer_func(self.model.parameters())
        if self.optimizer_reload_state is not None:
            self.optimizer.load_state_dict(self.optimizer_reload_state)
        self.scheduler = self.scheduler_func(self.optimizer)
        if self.scheduler_reload_state is not None:
            self.scheduler.load_state_dict(self.scheduler_reload_state)
        # Reset data loader to iterate over batches
        self.reset_data_loader(self.seed)
        self.data_iter = iter(self.train_loader)

        # Return the first batch as observations
        self.current_batch, self.current_labels = next(self.data_iter)
        return {"image": self.current_batch.cpu(), "array": self.current_labels.cpu()}#self.current_batch.cpu(),self.current_labels.cpu()


    def step(self, action):
        
        # increment counter
        self.count+=1

        # The action is the auxiliary task output generated by the aux_model for the current batch
        inputs, labels = self.current_batch.to(self.device),self.current_labels.to(self.device)

        # Forward pass for the main task model
        self.optimizer.zero_grad()
        class_output, aux_output = self.model(inputs)

        # define max softmax as per MAXL paper
        def mask_softmax( x, mask, dim=1):
            logits = torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True)
            return logits
        
        # create the mask
        def create_mask_from_labels(labels, num_classes=20, num_features=100):
            # Ensure num_features is a multiple of num_classes for correct partitioning
            assert num_features % num_classes == 0, "num_features must be a multiple of num_classes"
            
            ones_per_class = num_features // num_classes  # This is 5 in your case
            
            # Initialize an empty tensor of zeros with shape (N, num_features)
            N = len(labels)
            output_tensor = torch.zeros(N, num_features)
            
            for i, label in enumerate(labels):
                start_idx = label * ones_per_class
                end_idx = start_idx + ones_per_class
                output_tensor[i, start_idx:end_idx] = 1
            
            return output_tensor


        mask=create_mask_from_labels(labels).to(self.device)
        aux_target=mask_softmax(torch.tensor(action).to(self.device),mask,dim=-1)

        # Calculate classification loss and auxiliary loss
        #loss_class = self.criterion(class_output, labels)
        #loss_aux = self.criterion(aux_output, aux_target)
        loss_class   =  torch.mean(self.model.model_fit(nn.Softmax()(class_output), labels, pri=True,num_output=20))
        loss_aux  =  torch.mean(self.model.model_fit(nn.Softmax()(aux_output), aux_target,pri=False, num_output=100))

        
        info = {"loss_main" : loss_class.item(), "loss_aux": loss_aux.item() }
        # Combine losses (adjust aux_weight as needed)
        loss = loss_class + self.aux_weight * loss_aux
        loss.backward()
        self.optimizer.step()

        # Update observation to next batch
        try:
            self.current_batch, self.current_labels = next(self.data_iter)
            done = False
        except StopIteration:
            done = True
            if self.verbose:
                print("EPISODE FINISHED, steps: ",self.count)

        # Reward is based on reduction in loss, but classification accuracy also may make sense
        with torch.no_grad():
            # get random sample of dataset with replacement
            current_batch, current_labels = next(self.reward_sampler)
            inputs, labels = current_batch.to(self.device),current_labels.to(self.device)

            # get loss on updated model
            class_output, aux_output = self.model(inputs)
            loss_class_new = self.criterion(class_output, labels)
            reward =  - loss_class_new.item()
            entropy=0.2*torch.mean(self.model.model_entropy(aux_target))
            reward -= entropy
            self.return_+=reward
        
        # make sure batch size is correct. May be less on final iteration
        inputs, labels = self.current_batch, self.current_labels 
        while inputs.shape[0] < self.batch_size:
            inputs=torch.cat([inputs,inputs],dim=0)
            labels=torch.cat([labels,labels],dim=0)
        inputs=inputs[:self.batch_size]
        labels=labels[:self.batch_size]
        self.current_batch,self.current_labels  = inputs, labels
        
        return {"image": self.current_batch.cpu(), "array": self.current_labels.cpu()}, reward, done, info

    def render(self, mode='human'):
        pass  # Not needed for now

class LabelsFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(LabelsFeatureExtractor, self).__init__(observation_space, features_dim*64)
        
        self.embedding_layer = nn.Embedding(num_embeddings=100, embedding_dim=128)
        self.combine = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
    def forward(self, observations):
        labels= observations['array'].type(torch.LongTensor).to(device)  
        observations=observations['image']
        outs=[]
        for idx in range(observations.shape[0]):
            label_embeddings=self.embedding_layer(labels[idx])
            x=self.combine(label_embeddings)
            outs.append(x.flatten())
        outs= torch.stack(outs,dim=0)

        return outs
# Function to train the auxiliary network with RL
def get_ppo_agent( env, learning_rate=0.001,ent_coef=0.01,n_steps=79):
    # Set up the RL PPO agent (of course other agent types may make sense too)
    policy_kwargs = {
    "features_extractor_class": CustomFeatureExtractor, #CustomFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": 128},  # Dimensionality of the output features
    "net_arch":[],
    }  
    
    model = PPO("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs,batch_size=64, learning_rate=learning_rate,ent_coef=ent_coef,n_steps=n_steps,n_epochs=10)
    model.policy.action_net= ActionNet().cuda()
    model.policy.value_net=ValueNet().cuda()
    return model

# Function to train the auxiliary network with RL
def train_label_network_with_rl(model,env):
    episode_length = len(env.train_loader)  # Total steps in an episode
    model.learn(total_timesteps=episode_length)
    
def train_main_network(model, env):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Use deterministic greedy actions
        obs, reward, done, _info = env.step(action)
    env.update()
         

# Train
BATCH_SIZE=100
aux_dim=100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
total_epoch = 200
criterion = nn.CrossEntropyLoss()
model = VGG16([5]*20)
model.to(device)
optimizer_func = lambda x : optim.SGD(x, lr=0.01)
scheduler_func = lambda x : optim.lr_scheduler.StepLR(x, step_size=50, gamma=0.5)



env = AuxTaskEnv(cifar100_train_set, device,model,criterion, optimizer_func,scheduler_func, batch_size=BATCH_SIZE,aux_dim=aux_dim,aux_weight=1)
ppo_model = get_ppo_agent(env,learning_rate=0.0003,n_steps=79)




train_batch = len(cifar100_train_loader)
test_batch = len(cifar100_test_loader)
k = 0
avg_cost = np.zeros([total_epoch, 8], dtype=np.float32)
for index in range(total_epoch):
    model.train()
    train_label_network_with_rl(ppo_model,env)
    train_main_network(ppo_model, env)

    model=env.cannonical_model
    model.eval()
    # evaluating test data
    cost = np.zeros(4, dtype=np.float32)
    with torch.no_grad():
        cifar100_test_dataset = iter(cifar100_test_loader)
        for i in range(test_batch):
            test_data, test_label =  next(cifar100_test_dataset) #cifar100_test_dataset.next()
            test_label = test_label.type(torch.LongTensor)
            test_data, test_label = test_data.to(device), test_label.to(device)

            test_pred1, test_pred2 = model(test_data)
            test_pred1, test_pred2 = nn.Softmax()(test_pred1),nn.Softmax()(test_pred2)
            #print(ppo_model.policy({"image":test_data.unsqueeze(0), "array": test_label.unsqueeze(0)})[0].shape)
            aux_label = nn.Softmax()(ppo_model.policy({"image":test_data.unsqueeze(0), "array": test_label.unsqueeze(0)})[0].squeeze(0))
            
            test_loss1  = model.model_fit(test_pred1, test_label, pri=True,num_output=20)
            test_loss2  = model.model_fit(test_pred2, aux_label,pri=False, num_output=100)
            # evaluate on test data
            test_predict_label1 = test_pred1.data.max(1)[1]
            test_predict_label2 = test_pred2.data.max(1)[1]

            # calculate testing loss and accuracy for primary and auxiliary task
            test_acc1 = test_predict_label1.eq(test_label).sum().item() / batch_size
            test_acc2 = test_predict_label2.eq(aux_label).sum().item() / batch_size

            cost[0] = torch.mean(test_loss1).item()
            cost[1] = test_acc1
            cost[2] = torch.mean(test_loss2).item()
            cost[3] = test_acc2
            avg_cost[index][4:] += cost / test_batch

    print('EPOCH: {:04d} ITER: {:04d} | TRAIN [LOSS|ACC.]: PRI {:.4f} {:.4f} AUX {:.4f} {:.4f} || '
          'TEST [LOSS|ACC.]: PRI {:.4f} {:.4f} AUX {:.4f} {:.4f}'
          .format(index, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                  avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7]))
