{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code]\n# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here's several helpful packages to load\n\nimport numpy as np # linear algebra\nimport scipy\nimport scipy.linalg\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport torch\nimport torchvision\nimport torchvision.transforms as transforms\nimport pytorch_lightning as pl\nimport matplotlib.pyplot as plt\n\n# Input data files are available in the read-only \"../input/\" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session\n\n# %% [code]\n# original credits go to:\n# https://github.com/rfmiotto/CoSaMP/blob/master/cosamp.ipynb\ndef cosamp(Phi, u, s, tol=1e-10, max_iter=1000):\n    \"\"\"\n    @Brief:  \"CoSaMP: Iterative signal recovery from incomplete and inaccurate\n             samples\" by Deanna Needell & Joel Tropp\n\n    @Input:  Phi - Sampling matrix\n             u   - Noisy sample vector\n             s   - Sparsity vector\n\n    @Return: A s-sparse approximation \"a\" of the target signal\n    \"\"\"\n    max_iter -= 1 # Correct the while loop\n    num_precision = 1e-12\n    a = torch.Tensor( (Phi.shape[1]) )\n    v = u\n    iter = 0\n    halt = False\n    while not halt:\n        iter += 1\n#         print(\"Iteration {}\\r\".format(iter))\n#         print(v.shape)\n        y = abs(torch.matmul( (torch.transpose(Phi,0,1)), v))\n        Omega = [i for (i, val) in enumerate(y) if val > torch.sort(y)[::-1][2*s] and val > num_precision] # quivalent to below\n        #Omega = np.argwhere(y >= np.sort(y)[::-1][2*s] and y > num_precision)\n        T = np.union1d(Omega, a.nonzero()[0])\n        #T = np.union1d(Omega, T)\n        b = np.dot( np.linalg.pinv(Phi[:,T]), u )\n        igood = (abs(b) > np.sort(abs(b))[::-1][s]) & (abs(b) > num_precision)\n        T = T[igood]\n        a[T] = b[igood]\n        v = u - np.dot(Phi[:,T], b[igood])\n        \n        halt = np.linalg.norm(v)/np.linalg.norm(u) < tol or \\\n               iter > max_iter\n        \n    return a\n\n# %% [code]\ntrainset_full = torchvision.datasets.FashionMNIST(root=\"data\", train=True,\n                                             download=True, transform=transforms.ToTensor())\n\n# m_val = 50\n# N = 28*28\n# Random Rows Matrix\n# A = np.eye(N)\n# shuffled_vals = np.random.permutation(N)\n# A = A[shuffled_vals[:m_val],:]\n\n# %% [code]\ncompression_factor = 0.1\ncurr_seed = np.random.default_rng(seed=21) #Set RNG for repeatble results\n\nN =  28*28 #length of vectorized image\nm_val = int(compression_factor * N)\n\nA = curr_seed.standard_normal(((m_val), N)) #sensing matrix\nA = np.transpose(scipy.linalg.orth(np.transpose(A)))\nA = torch.from_numpy(A).type(torch.FloatTensor) \n\n# %% [code]\n\n\n# %% [code]\nims=trainset_full.data\nlabs = trainset_full.targets\nims_compressed = torch.Tensor(60,m_val)\nfor i in range(60): # np.size(ims,0)\n    temp = torch.reshape(ims[i,:,:],(28*28,)).type(torch.FloatTensor) \n    ims_compressed[i,:]= torch.matmul(A, temp)\n\n\n# %% [code]\nims_compressed[1,:].shape\n\n# %% [code]\ntorch.transpose(A,0,1)\n\n# %% [code]\ny = ims_compressed[0,:]\nidk = cosamp(A,y, 20) \nprint(np.size(idk))\n\n# %% [code]\ntest_im = np.reshape(idk,(28,28))\nplt.imshow(test_im)\n\n# %% [code]\nfor i in range(np.size(ims_compressed,0)):\n    y = ims_compressed[i,:]\n    idk = cosamp(A,y, 20) #?!?!?!?!!??!?!?!?!?!?!?!?!?!?!?!\n\n# %% [code]\nprint(np.size(ims,0))\n\n# %% [code]\nprint(labs[2])\nplt.imshow(ims[2,:,:])\n\n# %% [code]\n\n\n# %% [code]\nfunction recovered_ims = meas2recovery():\n\n\n# %% [code]\noutputs = ims_compressed[0,0:5]\nlabels = ims_compressed[1,0:5]\nprint(outputs)\nprint(labels)\nunion = (outputs | labels).float().sum((1, 2))\nprint(union)\n\n# %% [code]\n","metadata":{"_uuid":"94751193-b885-47dc-9a26-05692fc6c910","_cell_guid":"3f1f49e4-8a5c-479f-b06a-e8f7f3e4bd8e","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}