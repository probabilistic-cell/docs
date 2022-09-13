# Latenta development

## Connecting to the server

Our GPU server is at updeplasrv7.epfl.ch . You can only connect to this while connected to the EPFL VPN.
You should be able to connect in a terminal using `ssh USERNAME@updeplasrv7.epfl.ch`. In windows, your .ssh file is located here: https://superuser.com/questions/1544120/ssh-config-file-for-windows-10s-ssh
You can add the GPU server to your .ssh config by adding something the following to your `~/.ssh/config` 

```
Host gpu
    HostName updeplasrv7.epfl.ch
    User USERNAME
```

This allows you to connect using `ssh gpu`

It's highly adviseable to create and copy your ssh keys over to the server. On linux, you can do this using `ssh-copy-id gpu`. You may need to first create the ssh key pair using `ssh-keygen` (just keep pressing enter if this command ask you something). To do this on other operating systems, check out e.g. https://serverfault.com/questions/224810/is-there-an-equivalent-to-ssh-copy-id-for-windows https://stackoverflow.com/questions/25655450/how-do-you-install-ssh-copy-id-on-a-mac

## Development

We work with git as a version control system. We always do our work on a `devel` branch. Once a bunch of new features are added, we increase the version number and do a pull request on github to merge changes in the `main` branch.

We use github actions to automatically test most packages.

To develop locally, we typically use a conda environment that installs its own Python and packages. We then clone all the packages using e.g.:

```
git clone git@github.com/probabilistic-cell/latenta@devel
```

We then do `pip install -e ./latenta` to install the package in development mode. This means that all changes made to the package in the latenta folder will be immediately visible. If you then make changes in the latenta folder, or pull changes made by others, the package will be updated automatically without having to reinstall it.

## Interactive development

For running things, we typically use jupyter notebooks. However, these notebooks are never put in version control (i.e. in most .gitignore files you will see `*.ipynb`). Instead, we use jupytext to automatically sync these notebooks to a `.py` or `.md` file, which are text files and can be easily shared through git/github. In this way, we only version control the code itself, and never the results. These results should be shared on a server somewhere and then synced using e.g. rsync.

When you created a new notebook, you can activate the syncing with a py or md file by pressing ctrl+shift+c and move to "pair with ...".

## Using vscode

We use vscode as an IDE (https://code.visualstudio.com/). Please install it!

The gpu server has no jupyter lab server running. Instead, each user has to manage this themselves.
Within a conda environment (see later), you can run `jupyter lab`, which will start a jupyter server locally bound to localhost and a port, typically something around 8888. To forward this port to your machine, you could use the `ssh` command to forward a port, but I always use vs code to do this for me as discussed in the next paragraph.

We use the vscode "Remote Development" extension to to development on the gpu server, but with the vs code program running locally. To do this, you should install the extension and then do ctrl+shift+p -> connect to host -> `gpu`. Then you can add remote folders to the workspace, do source control, etc on the remote machine. Moreover, if you press ctrl+j and go to terminal, you get a remote terminal. If you then activate the conda environment and do jupyter lab, vs code will automatically forward to port for you (as shown in the ports tab). In the terminal, if you simply ctrl+mouse click on the url, you will be automatically redirected to the correct localhost port together with the password to access your jupyter lab.

It's highly adviseable to use a vs code workspace. To do this, simply save the workspace in the root directory of your project. Next time you open vs code, the workspace will be opened. Sometimes, vs code will also restart your jupyter lab server if you do this, although this is buggy.

## Github

It's highly adviseable, if not necessary, to copy your ssh keys to github. Otherwise you will constantly have to enter your github password. To do this, follow the steps at: https://docs.github.com/en/authentication/connecting-to-github-with-ssh. Note: you should generate these keys on the gpu!

Once I gave you access to the probabilistic-cell github organization, you should be able to access e.g. github.com/probabilistic-cell/latenta

## Conda

We use conda to manage our environments. You should install it in your account by following the instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

# Specifics to the project

## The folders

In your home directory, you can create "projects/probabilistic-cell", and clone the repositories "latenta", "laflow" and "lacell"

You should be able to create a vs code workspace, and add these folders to the workspace. You should also be able to see in the "source control" tab the four repositories associated to these folders. If you set up the ssh keys correctly, you should also be able to push and pull changes when pressing the "Synchronize changes" button. This is equivalent to running `git pull` and `git push`.

For a project, you will 

## Getting started

Note: I could have done these steps for you, but I think it's a great learning experience to do this yourself as well.)

Overview of linux commands: https://www.guru99.com/linux-commands-cheat-sheet.html

Set up the conda environment as follows:

```
conda create -n latenta

# activate your conda environment
conda activate latenta

# make sure you're in the projects/probabilistic-cell folder
conda install python=3.8

# install the correct pytorch version
# this is important to make sure we can work with a gpu
# as just doing conda install pytorch would install the cpu version
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

# install our packages in development mode
pip install -e ./latenta[full]
pip install -e ./laflow
pip install -e ./lacell

conda install jupyterlab
```

If you run the `jupyter lab` command while in the hca_spatial folder, then check whether the port is correctly forwarden in vs code, and then once your in the jupyter lab go to code/valentine, you will see one notebook there. Open it up and try to run it. If everything runs fine, hurray!

Now do ctrl+shift+c and select "pair with python script". If this option is not there you will need to install jupytext and restart the jupyter lab server. You can now try to make a change in this notebook and save it (press ctrl+s). This change should normally be reflected in the .py file association with the notebook. If you then check git's status in vs code (or run `git status`) you will see that the file is updated. Add the file by clicking on the plus, write a commit message, press ctrl+enter, synchronize the changes and you will see the file updated on github.