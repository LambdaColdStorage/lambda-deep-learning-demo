import os
import sys
from six.moves import urllib
import tarfile

def download_and_extract(data_file, data_url, create_parent_folder=True):
  data_dirname = os.path.dirname(data_file)
  print("Can not find " + data_file +
        ", download it now.")
  if not os.path.isdir(data_dirname):
    os.makedirs(data_dirname)

  if create_parent_folder:
    untar_dirname = data_dirname
  else:
    untar_dirname = os.path.abspath(os.path.join(data_dirname, os.pardir))

  download_tar_name = os.path.join("/tmp", os.path.basename(data_url))

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading to %s %.1f%%' % (
        download_tar_name, 100.0 * count * block_size / total_size))
    sys.stdout.flush()

  local_tar_name, _ = urllib.request.urlretrieve(data_url,
                                                 download_tar_name,
                                                 _progress)

  print("\nExtracting dataset to " + data_dirname)
  tarfile.open(local_tar_name, 'r:gz').extractall(untar_dirname)


def check_and_download(config):

  def check_meta_and_download(name_meta, flag_has_meta):
    if hasattr(config, name_meta):
      paths_meta = getattr(config, name_meta)

      if paths_meta:
        for path_meta in paths_meta:
          if path_meta:
            if not os.path.isfile(path_meta):
              download_and_extract(path_meta,
                                   config.dataset_url,
                                   False)
            else:
              print("Found " + path_meta + ".")
            flag_has_meta = True

    return flag_has_meta

  flag_has_meta = False

  flag_has_meta = check_meta_and_download('dataset_meta', flag_has_meta)
  flag_has_meta = check_meta_and_download('train_dataset_meta', flag_has_meta)
  flag_has_meta = check_meta_and_download('eval_dataset_meta', flag_has_meta)

  if not flag_has_meta and config.mode != "infer":
    assert False, "A meta data must be provided in non-inference mode." 