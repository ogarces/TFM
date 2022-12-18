import models
import torch.nn as nn
import segmentation_models_pytorch as smp

def ModelFactory(args):
    """with args defines depth and number of classes. Also initializes a model
    depending on the specified in args"""
    if args.task == 'mnist':
        args.depth = 1
        args.num_classes = 10

    if args.task == 'cifar10':
        args.depth = 3
        args.num_classes = 10

    if args.task == 'cifar100':
        args.depth = 3
        args.num_classes = 100
        args.size = args.img_width

    if args.task == 'tiny':
        args.depth = 4
        args.num_classes = 200
        args.size = 64

    if args.task == 'imagenet':
        args.num_classes = 1000
    
    if args.task == 'imagenet100':
        args.num_classes = 100
    
    if args.model == 'resnet20':
        train_net = models.resnet20(num_classes = args.num_classes)

    if args.model == 'resnet18':
        train_net = models.resnet18(num_classes = args.num_classes)

    if args.model == 'resnet50':
        train_net = models.resnet50(num_classes = args.num_classes)

    if args.model == 'resnet56':
        train_net = models.resnet56(num_classes = args.num_classes)

    if args.model == 'densenet121':
        train_net = models.densenet121(num_classes = args.num_classes)

    if args.model == 'lenet':
        if args.task == 'mnist':
            train_net = models.LeNetMNIST(num_classes = args.num_classes)
        else:
            train_net = models.LeNet(num_classes = args.num_classes)

    if args.model == 'convnet':
        train_net = models.ConvNet(3, args.num_classes, 128, args.depth, 'relu', 'instancenorm', 'avgpooling', im_size=(args.size, args.size))

    if args.model == 'alexnet':
        train_net = models.AlexNet(num_classes=args.num_classes)
        
        
############################################################################

        
    if args.task == 'iSAID':
        args.num_classes = 16
        
    if args.task =='potsdam':
        args.num_classes = 6
        
    if args.model == 'unet':
        train_net = models.UNet()
    
    if args.model =='segmentationresnet':
        train_net = models.segmentationResnet50(args.num_classes)
        # writer.add_graph(train_net, imgs)
        # writer.close()
        
    if args.model == 'deeplabv3':
        train_net = models.deeplabv3(args.num_classes)
 

    if args.model == 'lraspp':
        train_net = models.LRASPP_MOBILENET(7)
    
    if args.model == 'deeplabv3PlusR50':
        train_net = models.deeplabv3PlusR50(args.num_classes)
    
    
    
    return train_net
