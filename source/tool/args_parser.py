import os


def prepare(args):

  args.dataset_meta = os.path.expanduser(args.dataset_meta)
  args.model_dir = os.path.expanduser(args.model_dir)
  args.summary_names = (
    [] if not args.summary_names else
    args.summary_names.split(","))
  args.skip_pretrained_var_list = (
    [] if not args.skip_pretrained_var_list else
    args.skip_pretrained_var_list.split(","))
  args.skip_trainable_var_list = (
    [] if not args.skip_trainable_var_list else
    args.skip_trainable_var_list.split(","))
  args.trainable_var_list = (
    [] if not args.trainable_var_list else
    args.trainable_var_list.split(","))
  args.skip_l2_loss_vars = (
    [] if not args.skip_l2_loss_vars else
    args.skip_l2_loss_vars.split(","))
  args.augmenter = (None if not args.augmenter else
                    args.augmenter)
  return args
