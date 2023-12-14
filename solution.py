import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
import copy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.init_w = 3e-3
        layers = []

        layers.append(nn.Linear(self.input_dim, self.hidden_size))
        if self.activation == 'relu':
            layers.append(nn.ReLU())
        elif self.activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise NotImplementedError(f"Activation {activation} not implemented.")

        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())

        out_layer = nn.Linear(self.hidden_size, self.output_dim)
        out_layer.weight.data.uniform_(-self.init_w, self.init_w)
        out_layer.bias.data.uniform_(-self.init_w, self.init_w)
        layers.append(out_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        return self.model(s)
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.reparam_noise = 1e-6
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        self.actor_network = NeuralNetwork(self.state_dim, self.action_dim * 2, hidden_size=self.hidden_size,
                                            hidden_layers=self.hidden_layers, activation='relu').to(self.device)
        self.optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
    
    

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        mu_log_std = self.actor_network.forward(state)
        mu = mu_log_std[:,0]
        log_std = mu_log_std[:,1]
        log_std = self.clamp_log_std(log_std)
        sigma = torch.exp(log_std)
        normal = Normal(torch.unsqueeze(mu, 1), torch.unsqueeze(sigma, 1))

        if not deterministic:
            action_0 = normal.rsample()
        else:
            action_0 = torch.unsqueeze(mu, 1)
        action = torch.tanh(action_0)
        log_prob = normal.log_prob(action_0.to(self.device)) - torch.log(1 - action.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob
        

class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.critic_network = NeuralNetwork(self.state_dim + self.action_dim, 2, hidden_size=self.hidden_size,
                                            hidden_layers=self.hidden_layers, activation='relu').to(self.device)
        self.optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        self.gamma = 0.99
        self.alpha = 0.2
        self.tau = 0.005
        self.target_update_interval = 1
        self.hidden_layers = 3
        self.hidden_size = 256
        self.lr = 3e-4
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   
        self.policy_net = Actor(hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, actor_lr=self.lr, state_dim = self.state_dim, action_dim = self.action_dim, device = self.device)
        self.critic_1 = Critic(hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, critic_lr=self.lr, state_dim = self.state_dim, action_dim = self.action_dim, device = self.device) 
        self.critic_2 = Critic(hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, critic_lr=self.lr, state_dim = self.state_dim, action_dim = self.action_dim, device = self.device) 
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        #action = np.random.uniform(-1, 1, (1,))
        with torch.no_grad():
            state = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            action, _ = self.policy_net.get_action_and_log_prob(state, deterministic = not train)  
            action = action.squeeze(0)
        action  = action.detach().cpu().numpy()
        
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action[0]
    
    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        # This part of the code made use of the following github repo: 
        # https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/sac_pendulum.py

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        
        with torch.no_grad():
            next_action, next_log_prob = self.policy_net.get_action_and_log_prob(s_prime_batch, deterministic=False)
            qf1_next_target = self.critic_1_target.critic_network.forward(torch.cat([s_prime_batch, next_action], 1))
            qf2_next_target = self.critic_2_target.critic_network.forward(torch.cat([s_prime_batch, next_action], 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_prob
            next_q_value = r_batch + self.gamma * (min_qf_next_target)
        qf1 = self.critic_1.critic_network.forward(torch.cat([s_batch, a_batch], 1))
        qf2 = self.critic_2.critic_network.forward(torch.cat([s_batch, a_batch], 1)) 
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        qf2_loss = F.mse_loss(qf2, next_q_value) 
        self.run_gradient_update_step(self.critic_1, qf1_loss)
        self.run_gradient_update_step(self.critic_2, qf2_loss)

        next_action_pi, next_log_prob_pi = self.policy_net.get_action_and_log_prob(s_batch, deterministic=False)

        qf1_pi = self.critic_1.critic_network.forward(torch.cat([s_batch, next_action_pi], 1))
        qf2_pi = self.critic_2.critic_network.forward(torch.cat([s_batch, next_action_pi], 1))
        min_qf_policy = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * next_log_prob_pi) - min_qf_policy).mean()

        self.run_gradient_update_step(self.policy_net, policy_loss)

        self.critic_target_update(self.critic_1.critic_network, self.critic_1_target.critic_network, self.tau, soft_update=True)
        self.critic_target_update(self.critic_2.critic_network, self.critic_2_target.critic_network, self.tau, soft_update=True)

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    
    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = False
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
