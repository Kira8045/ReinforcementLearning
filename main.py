import random
import json
import argparse
import time
from drunkard import Drunkard

from dungeon import DungeonSimulator

from deepgambler import DeepGambler
from accountant import Accountant
from gambler import Gambler

def main():
    print("xxx")
    parser=argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default = 'DEEPGAMBLER',help="Which agent to use")
    parser.add_argument("--learning_agent", type=float, default = 0.1,help="Choose Learning Rate ")
    parser.add_argument("--discount", type=float, default = 0.95, help="choose discount")
    parser.add_argument("--iterations", type=int, default = 1000,help="No of iterations")

    FLAGS,unparsed = parser.parse_known_args()
    print("xxx")
    if FLAGS.agent == "DRUNKARD":
        agent= Drunkard()
    elif FLAGS.agent=="ACCOUNTANT":
        agent=Accountant()
    elif FLAGS.agent == "GAMBLER":
        agent=Gambler(FLAGS.learning_agent,FLAGS.discount,1.0,FLAGS.iterations)
    else:
        agent=DeepGambler(FLAGS.learning_agent,FLAGS.discount,1.0,FLAGS.iterations)
    dungeon= DungeonSimulator()

    dungeon.reset()
    total_reward = 0
    print("agent created")
    for step in range(FLAGS.iterations):
        old_state = dungeon.state
        action = agent.get_next_action(old_state)
        new_state , reward = dungeon.take_action(action)

        agent.update(old_state, new_state, action, reward)
        
        total_reward += reward

        if step %250 ==0: 
            print(json.dumps({'step': step,'total-reward':total_reward, }))

        time.sleep(0.0001)

    print("FINAL Q TABLE", agent.q_table)

if __name__ == "__main__":
    main()