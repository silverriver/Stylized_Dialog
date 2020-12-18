import torch


class MetricsRecorder(object):
    def __init__(self, device, *metric_names):
        self.metric_names = list(metric_names)
        self.metric_group_names = {n: n for n in metric_names}
        self.device = device
        self.metrics_value = {}
        self.metrics_count = {}
        for metric_name in metric_names:
            self.metrics_value[metric_name] = torch.tensor(0, dtype=torch.float64, device=self.device)
            self.metrics_count[metric_name] = torch.tensor(0, dtype=torch.float64, device=self.device)

    def metric_update(self, batch_no, *metric_values):
        for k, v in enumerate(metric_values):
            self.metrics_value[self.metric_names[k]] = self.metrics_value[self.metric_names[k]] + v[0]
            self.metrics_count[self.metric_names[k]] = self.metrics_count[self.metric_names[k]] + v[1]

    def add_permanent_metric(self, metric_name, value, count, metric_group=None):
        if metric_name not in self.metric_names:
            self.metric_names.append(metric_name)
        self.metrics_value[metric_name] = torch.tensor(value, dtype=torch.float64, device=self.device)
        self.metrics_count[metric_name] = torch.tensor(count, dtype=torch.float64, device=self.device)
        self.metric_group_names[metric_name] = metric_name if metric_group is None else metric_group

    def add_metric_groups(self, *metric_group_names):
        for i, n in enumerate(self.metric_names):
            self.metric_group_names[n] = metric_group_names[i]

    def add_to_writer(self, writer, step):
        for n in self.metric_names:
            value = self.metrics_value[n].item()
            count = self.metrics_count[n].item()
            if count == 0:
                writer.add_scalar('%s/%s' % (self.metric_group_names[n], n), 99999.9, step)
            else:
                writer.add_scalar('%s/%s' % (self.metric_group_names[n], n), value / count, step)


    def write_to_logger(self, logger, epoch, step):
        log_str = 'epoch {:>3}, step {}'.format(epoch, step)
        for n in self.metric_names:
            value = self.metrics_value[n].item()
            count = self.metrics_count[n].item()
            if count == 0:
                log_str += ', %s %g' % (n, 99999.9)
            else:
                log_str += ', %s %g' % (n, value / count)
        logger.info(log_str)

    def all_reduce(self):
        for n in self.metric_names:
            torch.distributed.all_reduce(self.metrics_value[n], op=torch.distributed.reduce_op.SUM)
            torch.distributed.all_reduce(self.metrics_count[n], op=torch.distributed.reduce_op.SUM)
