import configparser, sys
from semi.CNetTrain import Classifier, sutils

def export(config):
    c = Classifier.classifier(config)
    c.load(config.get("train", "output"))
    c.export(config.get("export","models"),
             config.get("export","dictionary"),
             config.get("export","config"))
    
def usage():
    print("usage:")
    print("\t python export.py config/eg.cfg")


if __name__ == '__main__':
    if len(sys.argv) != 2 :
        usage()
        sys.exit()
        
    config = configparser.ConfigParser(allow_no_value=True)
    try :
        config.read(sys.argv[1])
    except Exception as e:
        print("Failed to parse file")
        print(e)
    
    export(config)
