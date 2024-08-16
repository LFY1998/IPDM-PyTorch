from Config.default_config import default_cfg
from Utils.train_test_utils import progressive_domain_denoiser
if __name__ == '__main__':
    opt = default_cfg()
    model = progressive_domain_denoiser(opt)
    model.fit()
