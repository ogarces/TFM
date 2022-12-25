import os
import torch
import torch.nn as nn
from watchdog import WatchDog
import torch.distributed as dist
from dataloader import DataLoader
import torch.multiprocessing as mpgather_all
from arguments import ArgumentParser
from modelfactory import ModelFactory
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import *
from utils import get_optimizer, get_scheduler
import csv
import torch.multiprocessing as mp

# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_100A_0")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_25A_0") # b 14 #DONE
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_10A_0")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_5A_0") # natcj 20/10


# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_100A_1")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_25A_1") # b 14 #DONE
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_10A_1")
writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_5A_1")


# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_100A_2")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_25A_2") # b 14 #DONE

# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_10A_2")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_5A_2")

# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_100A_3")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_25A_3") # b 14
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_10A_3")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_5A_3")


# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_100A_4")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_25A_4") # b 14
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_10A_4")
# writer = SummaryWriter("experiments/DLV3Plr01wd00005focal_Pranc1_5A_4")




def gather_all_test(gpu_ind, args, train_net, testloader, criteria, e):
    print("gather all test started")
    c, t, tp, tn, fp, fn, val_mets = test(gpu_ind, args, train_net, testloader, criteria, e)
    total = torch.tensor([c, t], dtype=torch.float32, device=gpu_ind)
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    cnt, tot = total.tolist()
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
    f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="macro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    print("line 27 main_distrib", iou_score, f1_score, f2_score, accuracy, recall)
    print("gather all test ended")
    #print(tp)
    data = {'c':c, 't':t, 'tp': tp,  'tn':tn, 'fp':fp, 'fn':fn}

    
    
    


    return (cnt / tot) * 100, iou_score, f1_score, f2_score, accuracy, recall, data, val_mets
    


def main_worker( gpu_ind, args, shared_alpha):
    # Initializes the package for distributed training#function before calling any other 
    # methods. This blocks until all processes have joined.
    print("main worker started")
    rank = args.global_rank + gpu_ind       
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.world_size, 
        # between 0 and world_size -1      
    	rank=rank
    )
    print("initialization finished")
    # Loss function  cross entropy, mse, passed as arguments
    #if not args.loss == 'focal': 
    if not args.loss == 'focal':
        criteria = loss_func(args)
    if args.loss == 'focal':
        criteria = loss_func(args)
    print("Loss function is, " , criteria)
    criteria = criteria.to(gpu_ind)
    
    # Watchdog measures the start, end and time elapsed
    test_watchdog = WatchDog()
    
    # MODEL FACTORY with args defines depth and number of classes. Also initializes a model
    # depending on the specified in args
    print("instance of model")
    train_net = ModelFactory(args).to(gpu_ind)
    print("will initiate now")
    train_net = init_net(gpu_ind, args, train_net).to(gpu_ind)
    print("will do dataloader now")

    
    if args.task == 'segmentisaid':
        from dataloader import get_dataloadersisaid
        DATA_DIR = './datasets/iSAID'
        trainloader, testloader = get_dataloadersisaid(DATA_DIR,args.batch_size, args.batch_size)
    
    else:
        print('SECOND WAY')
        trainloader, testloader = DataLoader(args)
        
    

                
                
                

    torch.cuda.set_device(gpu_ind)
    train_net = DDP(train_net, device_ids=[gpu_ind])




            
            
    # normal training, not to be confused si pranc training
    if args.method == 'normal':
        max_acc = 0
        max_iou = 0
        maxF1 = 0
        maxF2 = 0
        maxacc = 0
        max_rec = 0
        val_mets = []
        train_mets = []
        f1s = []
        ious = []
        print("normal train")
        if args.resume is not None:
            print("normal train, resuming")
            # Loads a model’s parameter dictionary using a deserialized state_dict
            train_net.load_state_dict(load_model(gpu_ind, args))
            
        # acc, iou_score, f1_score, f2_score, accuracy, recall = gather_all_test(gpu_ind, args, train_net, testloader,criteria, e)
        acc, iou_score, f1_score, f2_score, accuracy, recall = 0, 0, 0, 0, 0, 0
        # sets learning rate
        optimizer = get_optimizer(args, train_net.parameters())
        # learning rate scheduler
        scheduler = get_scheduler(args, optimizer)
        # trains for n epochs
        print('will train for epochs')
        for e in range(args.epoch):
            normal_train_single_epoch(gpu_ind, args, e, train_net, trainloader, criteria, optimizer)
            if e % 1 == 0:
                test_watchdog.start()
                print('will calculate accuracy in gather all tests')
                acc, iou_score, f1_score, f2_score, accuracy, recall,data, val_met = gather_all_test(gpu_ind, args, train_net, testloader, criteria, e)
                val_mets.append(val_met)
                writer.add_scalar('iou', iou_score, e )
                writer.add_scalar('f1', f1_score, e )
                test_watchdog.stop()
                if gpu_ind == 0:
                    print("TEST RESULT:\tAcc:", round(acc, 3), 
                        "\tBest Acc:", max_acc, 
                        "\tIoU:", iou_score, "\tBest IoU:", max_iou,
                        "\tF2:", f2_score, "\tBest F2:", maxF2,
                        "\tF1:", f1_score, "\tBest F1:", maxF1,
                        "\accuracy:", accuracy,  "\tBest accuracy:", maxacc,
                        "\trecall:", recall, "\tBest recall:", max_rec,
                        "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
                    
                    

                    if not args.task == 'prancM':
                        if f1_score > maxF1 :
                            maxF1 = f1_score
                            save_model(gpu_ind, args, train_net) 
                            
                        if f2_score > maxF2:
                            maxF2 = f2_score
                            print('new max iou, printing iou, f1, f2, accuracy', max_iou, f1_score, f2_score, accuracy)

                        if iou_score > max_iou:
                            max_iou = iou_score
                            print('new max iou, printing iou, f1, f2, accuracy', max_iou, f1_score, f2_score, accuracy)
                                
                                   
                    else:
                        if acc > max_acc:
                            max_acc = acc
                            save_model(gpu_ind, args, train_net)    
                            

                    # if args.task == 'iSAID' and iou_score > max_iou:
                    #     max_iou = iou_score
                    #     save_model(gpu_ind, args, train_net)
                    # else: 
                    #     if acc > max_acc:
                    #         max_acc = acc
                    #         save_model(gpu_ind, args, train_net)
                    
                if args.scheduler == 'plateau':
                    scheduler.step(f1_score)
                else:                 
                    scheduler.step()


        if gpu_ind == 0: 
            if args.task == 'potsdam':
                print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3), iou_score)
            else:
                print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3), iou_score)
            


    if args.method == 'pranc':
        print('pranc training')
        max_acc = 0
        max_iou = 0
        maxF1 = 0
        maxF2 = 0
        maxacc = 0
        max_rec = 0
        val_mets = []
        train_mets = []
        f1s = []
        ious = []
        # pranc_init calls init_alpha with the data from training or zeroes, 
        # calls fill_basis_mat that sets a seed and then initializes weights for each model, each module
        # pranc_init uses no_grad
        alpha, basis_mat, train_net, train_net_shape_vec = pranc_init(gpu_ind, args, train_net)
        print("got alpha, basis mat, train net, train net shape vector, next will optimize alpha and he net")
        print("alpha", alpha, alpha.size() )
        print("basis_mat", basis_mat, basis_mat.size())
        # print(train_net)
        # print("train_net_shape_vec", train_net_shape_vec, train_net_shape_vec.size())
        
        if args.lr > 0:
            print('lr in args is greater than 0, gets alpha and net optimizer with pranc lr and lr and batchnorm')
            alpha_optimizer = get_optimizer(args, [alpha], 'pranc')
            net_optimizer = get_optimizer(args, train_net.parameters(), 'network')
            batchnorms = []
            # if the module is a batch normalization one, append it to batchnorms list
            # if we found at least one batchnorm module, then get an optimizer for it
            # if there is an optimizer for batch norm, then we get a scheduler
            for m in train_net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        batchnorms.append(p)
            if len(batchnorms) > 0:
                batchnorm_optimizer = get_optimizer(args, batchnorms, 'batchnorm')
            else:
                batchnorm_optimizer = None
            alpha_scheduler = get_scheduler(args, alpha_optimizer)
            if batchnorm_optimizer is not None:
                batchnorm_scheduler = get_scheduler(args, batchnorm_optimizer)
            else:
                batchnorm_scheduler = None
        else:
            alpha_scheduler = None
            batchnorm_scheduler = None
        print("main distrib gather all test to start")
        acc, iou_score, f1_score, f2_score, accuracy, recall = 0, 0, 0, 0, 0, 0
        print("main distrib gather all test ended")
        for e in range(args.epoch):
            train_met = pranc_train_single_epoch(gpu_ind, args, e, basis_mat, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer, batchnorm_optimizer)    
            train_mets.append(train_met)
            if e % 1 == 0 :
                test_watchdog.start()
                # print('line 198 main distrib start gather all test from pranc train')
                
                
                acc, iou_score, f1_score, f2_score, accuracy, recall, data , val_met = gather_all_test(gpu_ind, args, train_net, testloader, criteria, e)

                f1s.append(f1_score)
                ious.append(iou_score)
                val_mets.append(val_met)
                
                writer.add_scalar('iou', iou_score, e )
                writer.add_scalar('f1', f1_score, e )
                print('end gather all test from pranc train')
                test_watchdog.stop()
                
                if gpu_ind == 0:
                    print("TEST RESULT:\tAcc:", round(acc, 3), 
                        "\tBest Acc:", max_acc, 
                        "\tIoU:", iou_score, "\tBest IoU:", max_iou,
                        "\tF2:", f2_score, "\tBest F2:", maxF2,
                        "\tF1:", f1_score, "\tBest F1:", maxF1,
                        "\accuracy:", accuracy,  "\tBest accuracy:", maxacc,
                        "\trecall:", recall, "\tBest recall:", max_rec,
                        "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
                    
                    if args.task == 'potsdam':
                        if f1_score > maxF1:
                            maxF1 = f1_score
                            print("216 main distrib calculated fscore, saving model")
                            save_model(gpu_ind, args, train_net)      
                            print('217 main distrib, model saved') 
                            save_signature(gpu_ind, args, alpha, train_net, shared_alpha)  
                            print('219 main distrib signature saved')
                            max_acc = acc
                            print("221 utils saved model")
                            
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_100A_0.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_0.pkl', 'ab')) #DONE
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_0.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_0.pkl', 'ab'))


                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_100A_1.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_1.pkl', 'ab')) #DONE
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_1.pkl', 'ab'))
                            pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_1.pkl', 'ab'))
                            
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_100A_2.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_2.pkl', 'ab')) #DONE
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_2.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_2.pkl', 'ab'))
                            
                            
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_100A_3.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_3.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_3.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_3.pkl', 'ab'))  
                            
                            
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_100A_4.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_4.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_4.pkl', 'ab'))
                            # pickle.dump((args, e, data), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_4.pkl', 'ab'))
                            
                        if f2_score > maxF2:
                            maxF2 = f2_score
                            print('new max iou, printing iou, f1, f2, accuracy', max_iou, f1_score, f2_score, accuracy)

                        if iou_score > max_iou:
                            max_iou = iou_score
                            print('new max iou, printing iou, f1, f2, accuracy', max_iou, f1_score, f2_score, accuracy)

                        
                   
                    else:
                        if acc > max_acc:
                            max_acc = acc
                            save_model(gpu_ind, args, train_net)    
                            save_signature(gpu_ind, args, alpha, train_net, shared_alpha)  


                    # if args.task == 'iSAID' and iou_score > max_iou:
                    #     max_iou = iou_score
                    #     save_model(gpu_ind, args, train_net)
                    # else: 
                    #     if acc > max_acc:
                    #         max_acc = acc
                    #         save_model(gpu_ind, args, train_net)

            # print('VVV', train_mets, ious, f1s )  
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_0_metrics.pkl', 'ab')) #DONE
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_20A_0_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_15A_0_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_0_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_0_metrics.pkl', 'ab'))

            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_1_metrics.pkl', 'ab')) #DONE
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_20A_1_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_15A_1_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_1_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_1_metrics.pkl', 'ab'))
  
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_2_metrics.pkl', 'ab')) #DONE
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_20A_2_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_15A_2_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_2_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_2_metrics.pkl', 'ab'))
  
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_3_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_20A_3_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_15A_3_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_3_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_3_metrics.pkl', 'ab'))
            
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_25A_4_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_20A_4_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_15A_4_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_10A_4_metrics.pkl', 'ab'))
            # pickle.dump((args, e, train_mets, val_mets), open('/share/ogarces/PRANC/configs/stats/DLV3Plr01wd00005focal_Pranc1_5A_4_metrics.pkl', 'ab'))            
            
            
            alpha_scheduler.step()
            if batchnorm_scheduler is not None:
                batchnorm_scheduler.step()
        print("FINAL TEST RESULT:\tAcc:", max_acc)
    
    if args.method == 'pranc_bin':
        alpha, train_net, train_net_shape_vec, perm, perm_inverse = pranc_bin_init(gpu_ind, args, train_net)
        if args.lr > 0:
            alpha_optimizer = get_optimizer(args, [alpha], 'pranc')
            net_optimizer = get_optimizer(args, train_net.parameters(), 'network')
            batchnorms = []
            for m in train_net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        batchnorms.append(p)
            if len(batchnorms) > 0:
                batchnorm_optimizer = get_optimizer(args, batchnorms, 'batchnorm')
            else:
                batchnorm_optimizer = None
            alpha_scheduler = get_scheduler(args, alpha_optimizer)
            if batchnorm_optimizer is not None:
                batchnorm_scheduler = get_scheduler(args, batchnorm_optimizer)
            else:
                batchnorm_scheduler = None
        else:
            alpha_scheduler = None
            batchnorm_scheduler = None
        max_acc = gather_all_test(gpu_ind, args, train_net, testloader, criteria)
        for e in range(args.epoch):
            pranc_bin_train_single_epoch(gpu_ind, args, e, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer, perm, perm_inverse, batchnorm_optimizer)    
            if e % 1 == 0 :
                test_watchdog.start()
                acc = gather_all_test(gpu_ind, args, train_net, testloader, criteria)
                test_watchdog.stop()
                if gpu_ind == 0:
                    print("TEST RESULT:\tAcc:", round(acc, 3), "\tBest Acc:", round(max_acc,3), "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
                if acc > max_acc:
                    save_model(gpu_ind, args, train_net)
                    save_signature(gpu_ind, args, alpha, train_net, shared_alpha)             
                    max_acc = acc
            alpha_scheduler.step()
            if batchnorm_scheduler is not None:
                batchnorm_scheduler.step()
        print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3))

    if args.method == 'ppb': 
        alpha, train_net, train_net_shape_vec, perm, perm_inverse = ppb_init(gpu_ind, args, train_net)
        if args.lr > 0:
            alpha_optimizer = get_optimizer(args, [alpha], 'pranc')
            net_optimizer = get_optimizer(args, train_net.parameters(), 'network')
            batchnorms = []
            for m in train_net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        batchnorms.append(p)
            if len(batchnorms) > 0:
                batchnorm_optimizer = get_optimizer(args, batchnorms, 'batchnorm')
            else:
                batchnorm_optimizer = None
            alpha_scheduler = get_scheduler(args, alpha_optimizer)
            if batchnorm_optimizer is not None:
                batchnorm_scheduler = get_scheduler(args, batchnorm_optimizer)
            else:
                batchnorm_scheduler = None
        else:
            alpha_scheduler = None
            batchnorm_scheduler = None
        max_acc = gather_all_test(gpu_ind, args, train_net, testloader, criteria)
        for e in range(args.epoch):
            ppb_train_single_epoch(gpu_ind, args, e, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer, perm, perm_inverse, batchnorm_optimizer)    
            if e % 1 == 0 :
                test_watchdog.start()
                acc = gather_all_test(gpu_ind, args, train_net, testloader, criteria)
                test_watchdog.stop()
                if gpu_ind == 0:
                    print("TEST RESULT:\tAcc:", round(acc, 3), "\tBest Acc:", round(max_acc,3), "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
                if acc > max_acc:
                    save_model(gpu_ind, args, train_net)
                    save_signature(gpu_ind, args, alpha, train_net, shared_alpha)             
                    max_acc = acc
            alpha_scheduler.step()
            if batchnorm_scheduler is not None:
                batchnorm_scheduler.step()
        print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3))

    # if args.method == 'pranc_otf':
    #     alpha, train_net = pranc_otf_init(gpu_ind, args, train_net)
    #     if args.lr > 0:
    #         alpha_optimizer = get_optimizer(args, [alpha], 'pranc')
    #         net_optimizer = get_optimizer(args, train_net.parameters(), 'network')
    #         batchnorms = []
    #         for m in train_net.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 for p in m.parameters():
    #                     batchnorms.append(p)
    #         if len(batchnorms) > 0:
    #             batchnorm_optimizer = get_optimizer(args, batchnorms, 'batchnorm')
    #         else:
    #             batchnorm_optimizer = None
    #         alpha_scheduler = get_scheduler(args, alpha_optimizer)
    #         if batchnorm_optimizer is not None:
    #             batchnorm_scheduler = get_scheduler(args, batchnorm_optimizer)
    #         else:
    #             batchnorm_scheduler = None
    #     else:
    #         alpha_scheduler = None
    #         batchnorm_scheduler = None
        
    #     max_acc = gather_all_test(gpu_ind, args, train_net, testloader)
    #     for e in range(args.epoch):
    #         pranc_train_single_epoch(gpu_ind, args, e, basis_mat, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer, batchnorm_optimizer)    
    #         if e % 1 == 0 :
    #             test_watchdog.start()
    #             acc = gather_all_test(gpu_ind, args, train_net, testloader)
    #             test_watchdog.stop()
    #             if gpu_ind == 0:
    #                 print("TEST RESULT:\tAcc:", round(acc, 3), "\tBest Acc:", round(max_acc,3), "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
    #             if acc > max_acc:
    #                 save_model(gpu_ind, args, train_net)
    #                 save_signature(gpu_ind, args, alpha, train_net, shared_alpha)             
    #                 max_acc = acc
    #         alpha_scheduler.step()
    #         if batchnorm_scheduler is not None:
    #             batchnorm_scheduler.step()
    #     print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3))

if __name__ == '__main__':
    number_of_gpus = torch.cuda.device_count()
    max_acc = 0
    args = ArgumentParser()
    os.environ['MASTER_ADDR'] = args.dist_addr
    os.environ['MASTER_PORT'] = str(args.dist_port)
    
    if args.method == 'pranc':
        assert args.num_alpha % args.world_size == 0
        shared_alpha = torch.zeros(args.num_alpha)
        shared_alpha.share_memory_()
    else:
        shared_alpha = None
    mp.spawn(main_worker, nprocs = number_of_gpus, args=(args, shared_alpha))
    pass
