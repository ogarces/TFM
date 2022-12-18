import os
import pdb
import math
import torch
import random
import torch.optim 
import torch.nn as nn
from tqdm import tqdm
from watchdog import WatchDog
import torch.distributed as dist
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchmetrics
import segmentation_models_pytorch as smp

def prancable(m):
    return isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)

def has_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            return True
    return False

def save_signature(gpu_ind, args, alpha, train_net, shared_alpha):      
    if args.method == 'pranc_bin' or args.method == 'ppb':
        if gpu_ind != 0:
            return
        if os.path.isdir(args.task_id + '/' + args.save_path ) is False:
            os.mkdir(args.task_id + '/' + args.save_path)
        torch.save(alpha, args.task_id + '/' + args.save_path + '/alpha.pt')
        if has_batchnorm(train_net):         
            mean = []
            var = []
            bnw = []
            bnb = []
            for p in train_net.modules():
                if isinstance(p, nn.BatchNorm2d):
                    mean.append(p.running_mean)
                    var.append(p.running_var)
                    bnw.append(p.weight)
                    bnb.append(p.bias)
            
            torch.save(torch.cat(bnw), args.task_id + '/' + args.save_path +  '/bnw.pt')
            torch.save(torch.cat(bnb), args.task_id + '/' + args.save_path +  '/bnb.pt')
            torch.save(torch.cat(mean), args.task_id + '/' + args.save_path +  '/means.pt')
            torch.save(torch.cat(var), args.task_id + '/' + args.save_path +  '/vars.pt')
        return
        
    length = args.num_alpha // args.world_size
    start = length * gpu_ind
    end = start + length
    with torch.no_grad():
        shared_alpha[start:end].copy_(alpha)
    dist.barrier()
    if gpu_ind != 0:
        return
    if os.path.isdir(args.task_id + '/' + args.save_path ) is False:
        os.mkdir(args.task_id + '/' + args.save_path)
    torch.save(shared_alpha, args.task_id + '/' + args.save_path + '/alpha.pt')
    if has_batchnorm(train_net):
        mean = []
        var = []
        bnw = []
        bnb = []
        for p in train_net.modules():
            if isinstance(p, nn.BatchNorm2d):
                mean.append(p.running_mean)
                var.append(p.running_var)
                bnw.append(p.weight)
                bnb.append(p.bias)
        
        torch.save(torch.cat(bnw), args.task_id + '/' + args.save_path +  '/bnw.pt')
        torch.save(torch.cat(bnb), args.task_id + '/' + args.save_path +  '/bnb.pt')
        torch.save(torch.cat(mean), args.task_id + '/' + args.save_path +  '/means.pt')
        torch.save(torch.cat(var), args.task_id + '/' + args.save_path +  '/vars.pt')

def init_alpha(gpu_ind, args):
    # if there was training, loads that alpha, else initializes with zeroes
    if gpu_ind == 0:
        print("Initializing Alpha")
    length = args.num_alpha // args.world_size
    start = length * gpu_ind
    end = start + length
    print("from init_alpha length, start, end ", length, start, end)
    if args.resume is not None:
        alp = torch.load(args.resume + '/alpha.pt')[start:end]
        alp = alp.to(gpu_ind)
    else:
        alp = torch.zeros(length, requires_grad=True, device=torch.device(gpu_ind))
        with torch.no_grad():
            if gpu_ind == 0:
                alp[0] = 1.
    print('init alpha, initialized alpha, alfha is ', alp)
    return alp

def loss_func(args):
    if args.loss == 'mse':
        return nn.MSELoss()
    if args.loss == 'cross-entropy':
        return nn.CrossEntropyLoss()
    if args.loss == 'focal':
        return smp.losses.FocalLoss('multilabel')

    
def init_net(gpu_ind, args, train_net):
    if args.seed is not None:
        print("FROM init_net the seed was ", args.seed, " if seed, torch.cuda.manual_seed, else if has reset parameters, reset")
        if gpu_ind == 0:
            print("Initializing network with seed:", args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        if gpu_ind == 0:
            print("Initializing network with no seed")
    for p in train_net.modules():
        if hasattr(p, 'reset_parameters'):
            p.reset_parameters()
    return train_net

def get_optimizer(args, params, for_what='network'):
    lr = 0
    if for_what == 'network':
        print("getting lr for network")
        lr = args.lr
    if for_what == 'pranc':
        print("getting lr for pranc lr ", args.pranc_lr)
        lr = args.pranc_lr
    if for_what == 'batchnorm':
        print("getting lr for batchnorm", args.pranc_lr)
        lr = args.pranc_lr
    
    if args.optimizer == 'sgd':
        print("optimizer sgd")
        return torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        print("optimizer adam")
        return torch.optim.Adam(params, lr=lr)

def get_scheduler(args, optimzer):
    # returns the learning rate scheduler
    if args.scheduler == 'none':
        return torch.optim.lr_scheduler.StepLR(optimzer, 1,1)
    if args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimzer, args.scheduler_step, args.scheduler_gamma)
    if args.scheduler == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimzer, args.scheduler_gamma)

def normal_train_single_epoch(gpu_ind, args, epoch, train_net, trainloader, criteria, optimizer):
    train_net.train()
    train_watchdog = WatchDog()
    for batch_idx, data in enumerate(trainloader):
        train_watchdog.start()
        optimizer.zero_grad()
        imgs, labels = data
        imgs = imgs.to(gpu_ind)
        labels = labels.to(gpu_ind)
        loss = criteria(train_net(imgs), labels)
        loss.backward()
        optimizer.step()
        train_watchdog.stop()
        if batch_idx % args.log_rate == 0 and gpu_ind == 0:
            print("Epoch:", epoch, "\tIteration:", batch_idx, "\tLoss:", round(loss.item(), 4), "\tTime:", train_watchdog.get_time_in_ms(), 'ms')

def save_model(gpu_ind, args, train_net):
    if gpu_ind != 0:
        return 
    if os.path.isdir(args.task_id) is False:
        os.mkdir(args.task_id)
    torch.save(train_net.state_dict(), args.task_id + '/' + args.save_model)

def load_model(gpu_ind, args):
    return torch.load(args.resume, map_location=torch.device(gpu_ind))

def fill_basis_mat(gpu_ind, args, train_net):
    print("started fill basis mat, take each prancable layer of each net, flattens its parameters and saves")
    params = []
    for i, m in enumerate(train_net.modules()):
        if prancable(m):
            for p in m.parameters():
                params.append(p.flatten().shape[0])
                #print("FROM fill basis mat ", i,  "module, the train net module is ", m, " flattened params for basis mat ", p.flatten().shape[0], p.shape)
    cnt_param = sum(params)
    length = args.num_alpha // args.world_size
    print("length is ", length, " gpu ind is ", gpu_ind)
    start = length * gpu_ind
    print("start is ", start)
    end = start + length
    print("end is ", end)
    this_device = torch.device(gpu_ind)
    basis_mat = torch.zeros(length, cnt_param, device=this_device)
    print("FROM fill basis mat, basis mat of zeroes length x count params ", basis_mat.shape)
    if gpu_ind == 0:
        print("Initializing Basis Matrix:", list(basis_mat.shape))
    #print("\nFROM fill basis mat: for the length (alpha/world size), and all params of prancable layers then for modules in train_net if param shape is over, 2, under2, incialize dif ways ", p.shape)
    #print("BASIS mat is REcalculated, changed with each iteration of num models and params in prancable layers of each model")
    for i in tqdm(range(length)):
        torch.cuda.set_device(this_device)
        torch.cuda.manual_seed(i + start)
        #print("FROM fill basis mat set manual seed ", i+start)

        start_ind = 0
        print(i, "length, FROM fill basis mat, for each prancable layer of the module, check shape, if it is over 2, kaiming univorm")
        print("Start ind initially zero, ")
        for ii, m in enumerate(train_net.modules()):

            if prancable(m):
                #print(ii,"th module, working on module ", m)
                
                for j, p in enumerate(m.parameters()):
                    if len(p.shape) > 2:
                        t = torch.zeros(p.shape, device=this_device)
                        torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5))
                        basis_mat[i][start_ind:start_ind + t.flatten().shape[0]] = t.flatten()
                        start_ind +=  t.flatten().shape[0]
                        bound = 1 / math.sqrt(p.shape[1] * p.shape[2] * p.shape[3])
                        #print(ii, j, "module param, starting kaiming uniform, p.shape was greater than2 bound was ", bound, " start ind was ", start_ind)

                    if len(p.shape) == 2:
                        bound = 1 / math.sqrt(p.shape[1])
                        t = torch.zeros(p.shape, device=this_device)
                        torch.nn.init.uniform_(t, -bound, bound)
                        basis_mat[i][start_ind:start_ind + t.flatten().shape[0]] = t.flatten()
                        start_ind +=  t.flatten().shape[0]
                        #print(i, j, "module param,  starting uniform, p.shape was greater than2, bound was 2", bound, " start ind was ", start_ind)
                        
                    if len(p.shape) < 2:
                        t = torch.zeros(p.shape, device=this_device)
                        torch.nn.init.uniform_(t , -bound, bound)
                        basis_mat[i][start_ind:start_ind + t.flatten().shape[0]] = t.flatten()
                        start_ind +=  t.flatten().shape[0]
                        print(i, j, "module param, starting  uniform, p.shape was less than2 ound was ", bound, " start ind was ", start_ind)

    print("the basis mat is in the end ", basis_mat, "with size ", basis_mat.size())    
    return basis_mat

def pranc_init(gpu_ind, args, train_net):

    if gpu_ind == 0:
        print("Initializing PRANC")
    alpha = init_alpha(gpu_ind, args)
    basis_mat = fill_basis_mat(gpu_ind, args, train_net)
    train_net_shape_vec = torch.zeros(basis_mat.shape[1], device=basis_mat.device)
    print("FROM pranc init alpha shape, basis mat shape, train net shape vec shape " , alpha.shape, basis_mat.shape, train_net_shape_vec.shape)
    with torch.no_grad():
        # no grad disables gradient calculation, useful for inference
        start_ind = 0
        # matmul Matrix product of two tensors.
        # The behavior depends on the dimensionality of the tensors
        init_net_weights = torch.matmul(alpha, basis_mat).float()
        dist.all_reduce(init_net_weights, dist.ReduceOp.SUM, async_op=False)
        for m in train_net.modules():
            # prancable is if is fully connected or convolutional
            
            #print("FROM pranc_init, for each module if is prancable p.copy_(init_net_weights[start_ind:start_ind + p.flatten().shape[0]].reshape(p.shape))")
            if prancable(m):
                for p in m.parameters():
                    p.copy_(init_net_weights[start_ind:start_ind + p.flatten().shape[0]].reshape(p.shape))
                    start_ind +=  p.flatten().shape[0]
    if args.resume is not None:     
        if has_batchnorm(train_net):    
            means = torch.load(args.resume + '/means.pt', map_location=torch.device(gpu_ind))
            vars = torch.load(args.resume + '/vars.pt', map_location = torch.device(gpu_ind))
            bn_weight = torch.load(args.resume + '/bnw.pt', map_location=torch.device(gpu_ind))
            bn_bias = torch.load(args.resume + '/bnb.pt', map_location=torch.device(gpu_ind))
        ind = 0
        with torch.no_grad():
            # no grad disables gradient calculation, useful for inference
            for p1 in train_net.modules():
                if isinstance(p1, nn.BatchNorm2d):
                    leng = p1.running_var.shape[0]
                    p1.weight.copy_(bn_weight[ind:ind + leng])
                    p1.bias.copy_(bn_bias[ind:ind + leng])
                    p1.running_mean.copy_(means[ind:ind + leng])
                    p1.running_var.copy_(vars[ind:ind + leng])
                    ind += leng
    return alpha, basis_mat, train_net, train_net_shape_vec

def get_train_net_grads(train_net, train_net_grad_vec):
    with torch.no_grad():
        start_ind = 0
        for i, m in enumerate(train_net.modules()):
            if prancable(m):
                for p in m.parameters():
                    length = p.flatten().shape[0]
                    train_net_grad_vec[start_ind:start_ind + length] = p.grad.flatten()
                    start_ind += length
        return train_net_grad_vec

def update_train_net(alpha, basis_mat, train_net, train_net_shape_vec):
    train_net_shape_vec = torch.matmul(alpha, basis_mat).float()
    dist.all_reduce(train_net_shape_vec, dist.ReduceOp.SUM, async_op=False)
    with torch.no_grad():
        start_ind = 0
        for m in train_net.modules():
            if prancable(m):
                for p in m.parameters():
                    length = p.flatten().shape[0]
                    p.copy_(train_net_shape_vec[start_ind: start_ind + length].reshape(p.shape))
                    start_ind += length
        return train_net

def pranc_train_single_epoch(gpu_ind, args, epoch, basis_mat, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer, batchnorm_optimizer):
    train_net.train()
    train_watchdog = WatchDog()
    #print("FOR each item in Trainloeader, multiplied train net shape vec with basis mat to get alpha.grad")

    for batch_idx, data in enumerate(trainloader):
        
        train_watchdog.start()
        net_optimizer.zero_grad()
        alpha_optimizer.zero_grad()
        
        if batchnorm_optimizer is not None:
            batchnorm_optimizer.zero_grad()
            
        if args.task == 'iSAID':
            imgs = data['hr']
            labels = data['lab_hr']
        else:    
            imgs, labels = data
            
        imgs = imgs.to(gpu_ind)
        labels = labels.to(gpu_ind) # UNCOMMENTS THIS
        
        num_images = 0
        num_images += imgs.size(0)
        

        if args.task == 'iSAID':
            if args.model == 'segmentationresnet50': 
                outputs = train_net(imgs)
                _,predicted = torch.max(outputs,1)
                loss = criteria(outputs, torch.argmax(labels, dim=1))
            else:
                outputs = train_net(imgs)['out']
                loss = criteria(outputs, torch.argmax(labels, dim=1))
 
        else: 
            loss = criteria(train_net(imgs), labels)
            
        loss.backward()
            
        #print('WILL DO get_train_net_grads')
        train_net_shape_vec = get_train_net_grads(train_net, train_net_shape_vec)
        #print('WILL DO torch.matmul(train_net_shape_vec, basis_mat.T).float()')
        alpha.grad = torch.matmul(train_net_shape_vec, basis_mat.T).float()
        alpha_optimizer.step()
        if batchnorm_optimizer is not None:
            batchnorm_optimizer.step()
        train_net = update_train_net(alpha, basis_mat, train_net, train_net_shape_vec)
        train_watchdog.stop()
        if batch_idx % args.log_rate == 0 and gpu_ind == 0:
            print("Epoch:", epoch, "\tIteration:", batch_idx, "\tLoss:", round(loss.item(), 4), "\tTime:", train_watchdog.get_time_in_ms(), 'ms')

def init_bin_alpha(gpu_ind, args, train_net):
    if gpu_ind == 0:
        print("Initializing Alpha", args.num_alpha)
    random.seed(args.seed)
    total_param = []
    for m in train_net.modules():
        if prancable(m):
            for p in m.parameters():
                total_param.append(p.flatten().shape[0])
    total_param = sum(total_param)
    required_param = math.ceil(total_param / args.num_alpha) * args.num_alpha
    if args.resume is not None:
        alp = torch.load(args.resume + '/alpha.pt', map_location='cuda:' + str(gpu_ind))
    else:
        torch.cuda.set_device(gpu_ind)
        torch.cuda.manual_seed(args.seed)
        alp = torch.randn(args.num_alpha, requires_grad=True, device=torch.device(gpu_ind))
        with torch.no_grad():
            alp /= 10
    net_weights = torch.zeros(required_param, device=gpu_ind)
    net_grad = torch.zeros(required_param, device=gpu_ind)
    permutation = list(range(required_param))
    random.shuffle(permutation)
    perm = torch.tensor(permutation).reshape(args.num_alpha, -1)
    perm_inverse = [0] * required_param
    for i in range(len(permutation)):
        perm_inverse[permutation[i]] = i // (required_param // args.num_alpha)

    perm_inverse = torch.tensor(perm_inverse)
    return  perm, perm_inverse, alp, net_weights, net_grad

def pranc_bin_init(gpu_ind, args, train_net):
    if gpu_ind == 0:
        print("Initializing Binary PRANC")
    perm, perm_inverse, alpha, init_net_weights, net_grads = init_bin_alpha(gpu_ind, args, train_net)
    with torch.no_grad():
        start_ind = 0
        init_net_weights.copy_(alpha[perm_inverse])
        for m in train_net.modules():
            if prancable(m):
                for p in m.parameters():
                    p.copy_(init_net_weights[start_ind:start_ind + p.flatten().shape[0]].reshape(p.shape))
                    start_ind +=  p.flatten().shape[0]
    if args.resume is not None:    
        if has_batchnorm(train_net):    
            means = torch.load(args.resume + '/means.pt', map_location=torch.device(gpu_ind))
            vars = torch.load(args.resume + '/vars.pt', map_location = torch.device(gpu_ind))
            bn_weight = torch.load(args.resume + '/bnw.pt', map_location=torch.device(gpu_ind))
            bn_bias = torch.load(args.resume + '/bnb.pt', map_location=torch.device(gpu_ind))
        ind = 0
        with torch.no_grad():   
            for p1 in train_net.modules():
                if isinstance(p1, nn.BatchNorm2d):
                    leng = p1.running_var.shape[0]
                    p1.weight.copy_(bn_weight[ind:ind + leng])
                    p1.bias.copy_(bn_bias[ind:ind + leng])
                    p1.running_mean.copy_(means[ind:ind + leng])
                    p1.running_var.copy_(vars[ind:ind + leng])
                    ind += leng
    return alpha, train_net, net_grads, perm, perm_inverse

def setup_net( train_net, train_net_shape_vec):
    with torch.no_grad():
        start_ind = 0
        for m in train_net.modules():
            if prancable(m):
                for p in m.parameters():
                    length = p.flatten().shape[0]
                    p.copy_(train_net_shape_vec[start_ind: start_ind + length].reshape(p.shape))
                    start_ind += length
        return train_net

def pranc_bin_train_single_epoch(gpu_ind, args, epoch, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer, perm, perm_inverse, batchnorm_optimizer):
    train_net.train()
    train_watchdog = WatchDog()
    for batch_idx, data in enumerate(trainloader):
        train_watchdog.start()
        net_optimizer.zero_grad()
        alpha_optimizer.zero_grad()
        if batchnorm_optimizer is not None:
            batchnorm_optimizer.zero_grad()
        with torch.no_grad():
            train_net_shape_vec.copy_(alpha[perm_inverse])
            train_net = setup_net(train_net, train_net_shape_vec)
        imgs, labels = data
        imgs = imgs.to(gpu_ind)
        labels = labels.to(gpu_ind)
        loss = criteria(train_net(imgs), labels)
        loss.backward()
        if alpha.grad is None:
            alpha.grad = torch.zeros(alpha.shape, device=alpha.device)
        with torch.no_grad():
            train_net_shape_vec.copy_(get_train_net_grads(train_net, train_net_shape_vec))
            alpha.grad.copy_(torch.sum(train_net_shape_vec[perm], dim=1))
        alpha_optimizer.step()
        if batchnorm_optimizer is not None:
            batchnorm_optimizer.step()
        train_watchdog.stop()
        if batch_idx % args.log_rate == 0 and gpu_ind == 0:
            print("Epoch:", epoch, "\tIteration:", batch_idx, "\tLoss:", round(loss.item(), 4), "\tTime:", train_watchdog.get_time_in_ms(), 'ms')

def init_ppb_alpha(gpu_ind, args, train_net):
    if gpu_ind == 0:
        print("Initializing Alpha", args.num_alpha)
    random.seed(args.seed)
    total_param = []
    for m in train_net.modules():
        if prancable(m):
            for p in m.parameters():
                total_param.append(p.flatten().shape[0])
    total_param = sum(total_param)
    required_param = math.ceil(total_param / args.num_alpha) * args.num_alpha
    if args.resume is not None:
        alp = torch.load(args.resume + '/alpha.pt', map_location='cuda:' + str(gpu_ind))
    else:
        torch.cuda.set_device(gpu_ind)
        torch.cuda.manual_seed(args.seed)
        alp = torch.randn((args.num_beta, args.num_alpha), requires_grad=True, device=torch.device(gpu_ind))
        with torch.no_grad():
            alp /= 10
    net_weights = torch.zeros(required_param, device=gpu_ind)
    net_grad = torch.zeros(required_param, device=gpu_ind)
    permutation = []
    for i in range(args.num_beta):
        tmp = list(range(required_param))
        random.shuffle(tmp)
        permutation.append(tmp)
    perm = torch.tensor(permutation).reshape(args.num_beta, args.num_alpha, -1)
    perm_inverse = []
    for i in range(args.num_beta):
        tmp = [0] * required_param
        for j in range(len(permutation[i])):
            tmp[permutation[i][j]] = j // (required_param // args.num_alpha)
        perm_inverse.append(tmp)
    perm_inverse = torch.tensor(perm_inverse)
    return  perm, perm_inverse, alp, net_weights, net_grad

def ppb_init(gpu_ind, args, train_net):
    if gpu_ind == 0:
        print("Initializing PPB")
    perm, perm_inverse, alpha, init_net_weights, net_grads = init_ppb_alpha(gpu_ind, args, train_net)
    with torch.no_grad():
        start_ind = 0
        tmp = [alpha[i][perm_inverse[i]] for i in range(args.num_beta)]
        init_net_weights.copy_(sum(tmp))
        for m in train_net.modules():
            if prancable(m):
                for p in m.parameters():
                    p.copy_(init_net_weights[start_ind:start_ind + p.flatten().shape[0]].reshape(p.shape))
                    start_ind +=  p.flatten().shape[0]
    if args.resume is not None:    
        if has_batchnorm(train_net):   
            means = torch.load(args.resume + '/means.pt', map_location=torch.device(gpu_ind))
            vars = torch.load(args.resume + '/vars.pt', map_location = torch.device(gpu_ind))
            bn_weight = torch.load(args.resume + '/bnw.pt', map_location=torch.device(gpu_ind))
            bn_bias = torch.load(args.resume + '/bnb.pt', map_location=torch.device(gpu_ind))
        ind = 0
        with torch.no_grad():   
            for p1 in train_net.modules():
                if isinstance(p1, nn.BatchNorm2d):
                    leng = p1.running_var.shape[0]
                    p1.weight.copy_(bn_weight[ind:ind + leng])
                    p1.bias.copy_(bn_bias[ind:ind + leng])
                    p1.running_mean.copy_(means[ind:ind + leng])
                    p1.running_var.copy_(vars[ind:ind + leng])
                    ind += leng
    return alpha, train_net, net_grads, perm, perm_inverse

def ppb_train_single_epoch(gpu_ind, args, epoch, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer, perm, perm_inverse, batchnorm_optimizer):
    train_net.train()
    train_watchdog = WatchDog()
    for batch_idx, data in enumerate(trainloader):
        train_watchdog.start()
        net_optimizer.zero_grad()
        alpha_optimizer.zero_grad()
        if batchnorm_optimizer is not None:
            batchnorm_optimizer.zero_grad()
        with torch.no_grad():
            tmp = [alpha[i][perm_inverse[i]] for i in range(args.num_beta)]
            train_net_shape_vec.copy_(sum(tmp))
            train_net = setup_net(train_net, train_net_shape_vec)
        imgs, labels = data
        imgs = imgs.to(gpu_ind)
        labels = labels.to(gpu_ind)
        loss = criteria(train_net(imgs), labels)
        loss.backward()
        if alpha.grad is None:
            alpha.grad = torch.zeros(alpha.shape, device=alpha.device)
        with torch.no_grad():
            train_net_shape_vec.copy_(get_train_net_grads(train_net, train_net_shape_vec))
            for i in range(args.num_beta):
                alpha.grad[i].copy_(torch.sum(train_net_shape_vec[perm[i]], dim=1))
        alpha_optimizer.step()
        if batchnorm_optimizer is not None:
            batchnorm_optimizer.step()
        train_watchdog.stop()
        if batch_idx % args.log_rate == 0 and gpu_ind == 0:
            print("Epoch:", epoch, "\tIteration:", batch_idx, "\tLoss:", round(loss.item(), 4), "\tTime:", train_watchdog.get_time_in_ms(), 'ms')

# def init_alpha_otf(gpu_ind, args):
#     if gpu_ind == 0:
#         print("Initializing Alpha")
#     length = args.num_alpha // args.world_size
#     start = length * gpu_ind
#     end = start + length
#     if args.resume is not None:
#         alp = torch.load(args.resume + '/alpha.pt')[start:end]
#         alp = alp.to(gpu_ind)
#     else:
#         alp = torch.zeros(length, requires_grad=True, device=torch.device(gpu_ind))
#         with torch.no_grad():
#             if gpu_ind == 0:
#                 alp[0] = 1.
#     return alp

# def pranc_otf_init(gpu_ind, args, train_net):
#     if gpu_ind == 0:
#         print("Initializing PRANC On the Fly")
#     alpha = init_alpha_otf(gpu_ind, args)
#     basis_mat = fill_basis_mat(gpu_ind, args, train_net)
#     train_net_shape_vec = torch.zeros(basis_mat.shape[1], device=basis_mat.device)

#     with torch.no_grad():
#         start_ind = 0
#         init_net_weights = torch.matmul(alpha, basis_mat).float()
#         dist.all_reduce(init_net_weights, dist.ReduceOp.SUM, async_op=False)
#         for m in train_net.modules():
#             if prancable(m):
#                 for p in m.parameters():
#                     p.copy_(init_net_weights[start_ind:start_ind + p.flatten().shape[0]].reshape(p.shape))
#                     start_ind +=  p.flatten().shape[0]
#     if args.resume is not None:     
#         if has_batchnorm(train_net):    
#             means = torch.load(args.resume + '/means.pt', map_location=torch.device(gpu_ind))
#             vars = torch.load(args.resume + '/vars.pt', map_location = torch.device(gpu_ind))
#             bn_weight = torch.load(args.resume + '/bnw.pt', map_location=torch.device(gpu_ind))
#             bn_bias = torch.load(args.resume + '/bnb.pt', map_location=torch.device(gpu_ind))
#         ind = 0
#         with torch.no_grad():
#             for p1 in train_net.modules():
#                 if isinstance(p1, nn.BatchNorm2d):
#                     leng = p1.running_var.shape[0]
#                     p1.weight.copy_(bn_weight[ind:ind + leng])
#                     p1.bias.copy_(bn_bias[ind:ind + leng])
#                     p1.running_mean.copy_(means[ind:ind + leng])
#                     p1.running_var.copy_(vars[ind:ind + leng])
#                     ind += leng
#     return alpha, train_net




def test(gpu_ind, args, train_net, testloader):
    print("started test from utils")
    train_net.eval()

    print(testloader)
    
    cnt = 0
    total = 0
    
    for i, data in enumerate(testloader, 0):
        
        if args.task == 'iSAID':
            inputs = data['hr']
            labels = data['lab_hr']
        else:
            inputs, labels = data    
        

        if args.task == 'segmentsaid' or args.task == 'iSAID':
            if args.model == 'segmentationresnet50':
                outputs = train_net(inputs.to(gpu_ind))
            else:
                outputs = train_net(inputs.to(gpu_ind))['out']
                
            labels = torch.argmax(labels, dim=1).to(gpu_ind)
        else: 
            outputs = train_net(inputs.to(gpu_ind))
            labels = labels.to(gpu_ind)
    
        #outputs = torch.argmax(outputs, dim=1)
        
        
        
        #############################################################
        ############################################################# FROM ANOTHER CODE https://github.com/zackdilan/Semantic-Segmentation-with-Unet/blob/0f92e027041b1515ae50dfea1202d882295c73f7/utils.py#L42
        
        if args.task == 'iSAID':
            _,predicted = torch.max(outputs, 1)
            total += labels.nelement()  # to get the total number of pixels in the batch
            cnt += predicted.eq(labels.data).sum().item()
   
            jaccard = torchmetrics.JaccardIndex(num_classes=16).to(gpu_ind)
            print(jaccard(predicted, labels))

        else: 
            outputs = torch.argmax(outputs, dim=1)
            cnt += torch.sum(labels == outputs)
            total += labels.shape[0]



    return cnt, total