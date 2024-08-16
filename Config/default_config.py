import argparse
import json
import os
import sys


def default_cfg(argv=None):
    parser = argparse.ArgumentParser('Default arguments for training of different domain denoiser')
    # section: train/test cfg
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='model ckpt save frequency')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='test_batch_size')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument("--init_lr", default=2e-4, type=float)
    parser.add_argument('--test_numbers', type=int, default=50,
                        help='number of samples for test, -1 means test all, 0 means no test')
    parser.add_argument("--mode", type=str, default='train_img',
                        help='train_img / test_img / train_proj / test_proj / test_prog')
    # run_name and model_name
    parser.add_argument('--run_name', type=str, default='default',
                        help='each run name')
    parser.add_argument('--model_name', type=str, default='IPDM',
                        help='the type of method')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device id')
    parser.add_argument('--convertor', type=str, default='TV',
                        help='domain convertor')
    parser.add_argument('--load_option_path', type=str, default=None,
                        help='json options for loading')
    parser.add_argument('--load_img_model_path', type=str, default=None,
                        help='model params for loading')
    parser.add_argument('--load_proj_model_path', type=str, default=None,
                        help='model params for loading')
    parser.add_argument('--resume_epochs_proj', type=int, default=0,
                        help='number of epochs of proj model params for resuming')
    parser.add_argument('--resume_epochs_img', type=int, default=0,
                        help='number of epochs of img model params for resuming')
    parser.add_argument('--display_result', type=bool, default=False,
                        help='save figures of results')
    parser.add_argument('--test_result_data_save', type=bool, default=False,
                        help='save the data of test results')
    parser.add_argument('--benchmark_test', type=bool, default=False,
                        help='test FBP/TV/ART')
    parser.add_argument('--metrics', nargs='+', type=str, default=['psnr', 'ssim', 'fsim', 'vif', 'nqm'],
                        help='metrics for test')
    parser.add_argument('--fbp_sharpen', type=bool, default=False,
                        help='sharpen the result of the fbp')
    parser.add_argument('--ntv', type=int, default=0,
                        help='the number of TV')
    parser.add_argument('--normal', type=bool, default=False,
                        help='normalize the data for training')
    parser.add_argument('--ultra_img_denoise', type=bool, default=True,
                        help='ultra img domain denoise step for visual effect')


    # section: img model cfg for test
    parser.add_argument('--in_channels_img', type=int, default=1,
                        help='the input channels')
    parser.add_argument('--out_channels_img', type=int, default=1,
                        help='the output channels')
    parser.add_argument('--model_channels_img', type=int, default=64,
                        help='the base channels of the network')
    parser.add_argument('--attention_resolutions_img', nargs='+', type=int, default=[16],
                        help='the resolutions which need to be applied attention')
    # ch_m第一个是预卷积，不进行下采样，对于正常分辨率可以设置成1，mult应该逐渐增大
    # attention_resolutions的值为下采样倍数，对于channel_mult=[c1,c2,c3,c4,c5]来说，对应的attention_resolutions值为[1,2,4,8,8],因为最后一个channel_mult不下采样
    parser.add_argument('--channel_mult_img', nargs='+', type=float, default=[1, 1, 2, 2, 4, 4],
                        help='the channel times coefficient of the network')
    parser.add_argument('--timesteps_img', type=int, default=1000,
                        help='timesteps T of img domain')
    parser.add_argument('--partial_timesteps_img', type=int, default=50,
                        help='partial timesteps K of img domain for training')
    parser.add_argument('--schedule_power_img', type=float, default=1,
                        help='power of the beta schedule')
    parser.add_argument('--clip_img', type=bool, default=True,
                        help='clip to [0,1] in the reverse process')
    parser.add_argument('--save_states_img', type=bool, default=False,
                        help='save reverse states')
    parser.add_argument('--lambda_ratio_img', type=float, default=5,
                        help='pre-denoise lambda')
    parser.add_argument('--t_start_img', nargs='+', default=None, type=int,
                        help='partial timestep K for test')
    parser.add_argument('--eta_img', type=float, default=0.5,
                        help='update weight of the ldct during iteration')
    parser.add_argument('--constant_guidance_img', type=float, default=None,
                        help='constant value of the lambda in img domain')
    parser.add_argument('--kernel_size_img', type=int, default=4,
                        help='kernel size of the down sampling of the guidance')
    parser.add_argument('--amplitude_img', type=float, default=20,
                        help='amplitude of the exp of the guidance')
    parser.add_argument('--ddim_timesteps_img', nargs='+', type=int, default=[1, 2, 2],
                        help='if using the sparse sampling method, it need to set the ddim timesteps')
    parser.add_argument('--sample_method_img', type=str, default="dense",
                        help='sample method of the guided reverse process')
    parser.add_argument('--save_it_state_img', type=bool, default=False,
                        help='save iteration states')

    # section: projection model cfg for test
    parser.add_argument('--in_channels_proj', type=int, default=1,
                        help='the input channels')
    parser.add_argument('--out_channels_proj', type=int, default=1,
                        help='the output channels')
    parser.add_argument('--model_channels_proj', type=int, default=64,
                        help='the base channels of the network')
    parser.add_argument('--attention_resolutions_proj', nargs='+', type=int, default=[32],
                        help='the resolutions which need to be applied attention')
    parser.add_argument('--channel_mult_proj', nargs='+', type=float, default=[1 / 64, 2 / 64, 4 / 64, 2, 2, 4, 4],
                        help='the channel times coefficient of the network')
    parser.add_argument('--timesteps_proj', type=int, default=1000,
                        help='timesteps of projection domain for training')
    parser.add_argument('--partial_timesteps_proj', type=int, default=50,
                        help='partial timesteps K of img domain')
    parser.add_argument('--schedule_power_proj', type=float, default=1,
                        help='power of the beta schedule')
    parser.add_argument('--clip_proj', type=bool, default=False,
                        help='clip to [0,1] in the reverse process')
    parser.add_argument('--lambda_ratio_proj', type=float, default=5,
                        help='pre-denoise lambda')
    parser.add_argument('--t_start_proj', nargs='+', default=None, type=int,
                        help='partial timestep K for test')
    parser.add_argument('--eta_proj', type=float, default=0.4,
                        help='update weight of the ldct during iteration')
    parser.add_argument('--constant_guidance_proj', type=float, default=None,
                        help='constant value of the lambda in proj domain')
    parser.add_argument('--kernel_size_proj', type=int, default=4,
                        help='kernel size of the down sampling of the guidance')
    parser.add_argument('--amplitude_proj', type=float, default=5,
                        help='amplitude of the exp of the guidance')
    parser.add_argument('--ddim_timesteps_proj', nargs='+', type=int, default=[1, 2, 2],
                        help='if using the sparse sampling method, it need to set the ddim timesteps')
    parser.add_argument('--sample_method_proj', type=str, default="dense",
                        help='sample method of the guided reverse process')
    parser.add_argument('--save_it_state_proj', type=bool, default=False,
                        help='save iteration states')

    # section: dataset cfg
    parser.add_argument('--data_type', type=str, default="siemens")
    parser.add_argument('--train_dataset_path_FD_img', type=str, default=None)
    parser.add_argument('--train_dataset_path_LD_img', type=str, default=None)
    parser.add_argument('--train_dataset_path_FD_proj', type=str, default=None)
    parser.add_argument('--train_dataset_path_LD_proj', type=str, default=None)
    parser.add_argument('--test_dataset_path_FD_img', type=str, default=None)
    parser.add_argument('--test_dataset_path_LD_img', type=str, default=None)
    parser.add_argument('--test_dataset_path_FD_proj', type=str, default=None)
    parser.add_argument('--test_dataset_path_LD_proj', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='dataloader num_workers')
    parser.add_argument('--patch', nargs='+', type=int, default=[512, 512],
                        help='patch size for training')
    parser.add_argument('--patch_per_image', type=int, default=4,
                        help='number of patches of each image')
    parser.add_argument('--dose', type=float, default=0.25,
                        help='dose% data use for training and testing')
    if argv is not None:
        opt = parser.parse_args(argv)
    else:
        argv = sys.argv[1:]
        opt = parser.parse_args(argv)

    # 注意：如果使用option.json覆盖，会导致命令行输入的参数被覆盖
    # 如果想让输入优先级大于option覆盖，需要获取命令行输入参数，在加载时排除这些参数
    args_input = [item[2:] for item in argv if "--" in item]
    if opt.load_option_path is not None:
        print("options are loading...")
        print("loading cfg except {}".format(args_input))
        load_option(opt, opt.load_option_path, args_input)
        print("options were loaded successfully!")
    return opt


# new_cfg是新增修改的部分，old_cfg是需要修改值的
def cfg_load(new_cfg, old_cfg):
    for key in new_cfg.keys():
        if isinstance(new_cfg[key], dict):
            cfg_load(new_cfg[key], old_cfg[key])
        else:
            if key in old_cfg.keys():
                old_cfg[key] = new_cfg[key]
            else:
                print(f"no key names {key} in config\n")
                # sys.exit()


def load_option(opt, load_path, exception):
    f = open(load_path, 'r')
    opt_load = json.load(f)
    for key in exception:
        if key in opt_load.keys():
            del opt_load[key]
    cfg_load(opt_load, opt.__dict__)
