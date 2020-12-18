import torch


class MetricsRecorder(object):
    def __init__(self, device, *metric_names):
        self.metric_names = list(metric_names)
        self.metric_group_names = {n: n for n in metric_names}
        self.device = device
        self.metrics = {}
        for metric_name in metric_names:
            self.metrics[metric_name] = torch.tensor(0, dtype=torch.float64, device=self.device)

    def metric_update(self, batch_no, *metric_values):
        for k, v in enumerate(metric_values):
            self.metrics[self.metric_names[k]] = (self.metrics[self.metric_names[k]] * batch_no + v) / (batch_no + 1)

    def add_permanent_metric(self, metric_name, value, metric_group=None):
        if metric_name not in self.metric_names:
            self.metric_names.append(metric_name)
        self.metrics[metric_name] = torch.tensor(value, dtype=torch.float64, device=self.device)
        self.metric_group_names[metric_name] = metric_name if metric_group is None else metric_group

    def add_metric_groups(self, *metric_group_names):
        for i, n in enumerate(self.metric_names):
            self.metric_group_names[n] = metric_group_names[i]

    def add_to_writer(self, writer, step):
        for n in self.metric_names:
            m = self.metrics[n].item()
            writer.add_scalar('%s/%s' % (self.metric_group_names[n], n), m, step)

    def write_to_logger(self, logger, epoch, step):
        log_str = 'epoch {:>3}, step {}'.format(epoch, step)
        for n in self.metric_names:
            m = self.metrics[n].item()
            log_str += ', %s %g' % (n, m)
        logger.info(log_str)

    def all_reduce(self):
        for n in self.metric_names:
            torch.distributed.all_reduce(self.metrics[n], op=torch.distributed.ReduceOp.SUM)
            self.metrics[n] /= torch.distributed.get_world_size()
