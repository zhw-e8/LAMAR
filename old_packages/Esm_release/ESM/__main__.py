import fire
from ESM import load_trainer
def train(config_path,*args,**kwargs):
    print(args,kwargs)
    trainer=load_trainer(config_path,**kwargs)
    trainer.train(resume_from_checkpoint=True)
    
if __name__=="__main__":
    fire.Fire(train)