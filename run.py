from bot.simple_bot import Simple_bot
import sys 
import argparse
if __name__ =='__main__': 
    reload(sys) 
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser("Runs the chatbot in either training or test mode") 
    parser.add_argument("-t","--test", help="run in test mode", action="store_true")
    
    args = parser.parse_args() 
    print("---------------------------------------")
    if args.test:
        print("Mode: test") 
        print("Training please wait...") 
        bot1 = Simple_bot(0.7, True)
        print("Testing please wait...") 
        bot1.test()

    else:
        print("Mode: chat")
        print("Training please wait...") 
        bot1 = Simple_bot(1, False)
        print("Type a message and press enter to get a response") 
        bot1.chat()
