import numpy as np
import torch

class GNS:
    def __init__(self) -> None:
        pass

    def compute_grad(self, sample, target, model, loss_fn):
        sample = sample.unsqueeze(0)  # prepend batch dimension for processing
        target = target.unsqueeze(0)

        prediction = model(sample)
        loss = loss_fn(prediction, target)

        return torch.autograd.grad(loss, list(model.parameters()))


    def compute_sample_grads(self, data, targets, model, loss_fn):
        """ manually process each sample with per sample gradient """
        sample_grads = [self.compute_grad(data[i], targets[i], model, loss_fn) for i in range(len(data))]
        sample_grads = zip(*sample_grads)
        sample_grads = [torch.stack(shards) for shards in sample_grads]
        return sample_grads

    
    def compute_gns(self, sample_grads, type="vector"):
        '''
        split will caclucate gns for each parameters instead of regarding it as a whole
            [   
                [
                    [sample1's],
                    [sample2's]
                ], # tensor 1
                [
                    [sample1's],
                    [sample2's]
                ], # tensor 2
            ]
        '''
        num_samples = sample_grads[0].shape[0]
        gns_list = []
        layer_required = [0, 3, 6, 9]

        if type == "vector":
            for i in range(len(sample_grads)):
                if i not in layer_required:
                    continue
                                        
                tensor = sample_grads[i]
                temp = tensor.cpu().numpy()
                G = []
                for i in range(num_samples):
                    G.append(np.ravel(temp[i]))

                transpose = np.transpose(G)
                for w in transpose:
                    var = np.var(w)
                    avg = np.average(w)
                    gns = np.sqrt(var / np.power(avg, 2))
                    gns_list.append(gns)

        elif type == "split":
            print("total tensors: ", len(sample_grads))
            for i in range(len(sample_grads)):
                if i not in layer_required:
                    continue
                                        
                tensor = sample_grads[i]
                temp = tensor.cpu().numpy()
                G = []
                for i in range(num_samples):
                    G.append(np.ravel(temp[i]))

                '''
                    weight1: [s1, s2, ...]
                    weight2: [s1, s2, ...]
                '''
                transpose = np.transpose(G)
                tr = np.cov(transpose).trace()
                norm = np.linalg.norm(transpose, ord=2)
                gns = tr / norm
                gns_list.append(gns)

        else:
            length = 0
            for tensor in sample_grads:
                tensor_1dim = tensor[0].reshape(-1)
                length += tensor_1dim.shape[0]
            G = torch.ones([num_samples, length])

            for i in range(num_samples):
                for j in range(len(sample_grads)):
                    tensor = sample_grads[j]
                    if j == 0:
                        flat_tensor = tensor[i].reshape(-1)
                    else:
                        flat_tensor = torch.cat((flat_tensor, tensor[i].reshape(-1)), -1)
                
                G[i] = flat_tensor

            G = G.numpy()
            transpose = np.transpose(G)
            tr = np.cov(transpose).trace()
            norm = np.linalg.norm(transpose, ord=2)
            gns = tr / norm # https://torch-foresight.readthedocs.io/en/latest/modules/gns.html
            gns_list.append(gns)

        return np.mean(gns_list)
