import gfootball.env as football_env
import keras.backend as K
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np

def get_model_actor_image(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1,output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    feature_extractor = MobileNetV2(weights="imagenet", include_top=False)

    # nao treinar esse feature extractor
    for layer in feature_extractor.layers:
        layer.trainable = False

    x = Flatten(name="flatten")(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values], outputs=[out_actions])
    model.compile(optimizer = Adam(lr=1e-4), loss=[get_ppo_loss(oldpolicy_probs=oldpolicy_probs,advantages = advantages,
                                                                 rewards=rewards, values=values)])
    
    return model

def get_model_critic_image(input_dims):
    state_input = Input(shape=input_dims)
    feature_extractor = MobileNetV2(weights="imagenet", include_top=False)
    # nao treinar esse feature extractor
    for layer in feature_extractor.layers:
        layer.trainable = False

    x = Flatten(name="flatten")(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    #mudamos para apenas o output, 1 valor real que indica a qualidade do estado acessado
    out_actions = Dense(1, activation='tanh', name='predictions')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer = Adam(lr=1e-4), loss='ms   e')
    
    return model

def get_advantage(values, masks, rewards):
    returns = []
    gae = 0

    # seguindo a formula para gae
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma*values[i+1] *masks[i] - values[i]
        gae = delta + gamma*lambda_* masks[i]*gae

        return_i = gae + values[i]
        returns.insert(0, return_i)

    advantage = np.array(returns) - values[:1]
    # normalizando
    advantage = (advantage - np.mean(advantage))/(np.std(advantage) + 1e-10)

    return returns, advantage


def get_ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))

        p1 = ratio*advantages
        p2 = K.clip(ratio, min_value= 1 - clip_param, max_value= 1 + clip_param)

        actor_loss = -K.mean(K.minimum(p1,p2))
        critic_loss = K.mean(K.square(rewards-values))

        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(-(newpolicy_probs*K.log(newpolicy_probs + 1e-10)))
        
        return total_loss
    
    return loss

env = football_env.create_environment(env_name="academy_empty_goal", representation="pixels", render=True)

n_actions = env.action_space.n
state_dims = env.observation_space.shape

state = env.reset()

ppo_steps = 4
gamma = 0.99
lambda_ = 0.95
clip_param = 0.2
critic_discount = 0.5
entropy_beta = 0.001


actor_model = get_model_actor_image(input_dims=state_dims, output_dims=n_actions)
critic_model = get_model_critic_image(input_dims=state_dims)

states = []
actions = []
values = []
masks = [] # verifica se jogo terminou
rewards = []
actions_probs = []
actions_onehot = []

dummy_n = np.zeros((1,1,n_actions))
dummy_1 = np.zeros((1,1,1))

# Colentando experiencias
for i in range(ppo_steps):
    state_input = K.expand_dims(state, 0)

    # prever a distribuicao de ações (politica), e escolher uma acao da distribuicao
    action_dist = actor_model.predict([state_input, dummy_n, dummy_1,  dummy_1,  dummy_1], steps=1)
    action = np.random.choice(n_actions, p=action_dist[0, :])
    action_one_hot = np.zeros(n_actions)
    action_one_hot[action] = 1

    q_value = critic_model.predict([state_input], steps=1)

    obs, reward, done, info = env.step(env.action_space.sample())
    masks = not done

    # Registra
    states.append(state)
    actions.append(action)
    actions_onehot.append(action_one_hot)
    values.append(q_value)
    rewards.append(reward)
    actions_probs.append(action_dist)

    # altera o estado para a nova observação
    state = obs

    if done:
        env.reset()

# mais um registro pro q_value
state_input = K.expand_dims(state, 0)
q_value = critic_model.predict([state_input], steps=1)
values.append(q_value)

# calculando vantagens e retornos para o treinamento
returns, advantages = get_advantage(values, masks, rewards)
actor_model.fit([states, actions_probs, advantages, np.reshape(rewards, new_shape=(-1,1,1)), values[:-1]], 
                [np.reshape(actions_onehot, new_shape=(-1,n_actions))],
                verbose=True, shuffle=True, epochs=5)
critic_model.fit([states], [np.reshape(returns, new_shape=(-1,1))], verbose=True, shuffle=True, epochs=5)
env.close()