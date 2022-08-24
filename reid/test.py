import os

def safe_link(src, dst):
    if os.path.islink(dst):
        os.unlink(dst)
    os.symlink(src, dst)

working_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
source_dir = os.path.join(working_dir, 'eval/market2duke')
print(source_dir)
# safe_link('/home/zyb/projects/h-go/eval/market2duke-train/pid.txt', 'data/market_DukeMTMC-reID-train/renew_pid.log')