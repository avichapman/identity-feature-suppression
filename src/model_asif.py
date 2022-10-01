import math
import os
from enum import Enum
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import List, Tuple, Union, Iterator, Any, Optional

from torchvision.models import resnet101, resnet18
from model_idn_wideresnet import IdnWideResNet
from model_nishi_resnet import NishiResNet18
from model_wideresnet import WideResNet


# noinspection PyMethodOverriding
class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        return dx, None


class OriginalFeatureExtractorModel(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = 8

        features = [2, 'M', 4, 'M', self.out_channels]
        self.layers = self._make_layers(features, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        x = self.layers(x)
        x = functional.avg_pool2d(x, x.size()[2])
        x = x.view(batch_size, -1)
        return x

    @staticmethod
    def _make_layers(cfg: List[Union[int, str]], in_channels: int):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=(1, 1))
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class BaseModelType(Enum):
    """
    Possible types of base model.
    """
    BASE_MODEL_RESNET_18_PRETRAINED = 'pretrained_resnet18'
    BASE_MODEL_RESNET_18_UNTRAINED = 'untrained_resnet18'
    BASE_MODEL_RESNET_101 = 'resnet101'
    BASE_MODEL_WIDE_RESNET_101 = 'wresnet101'
    BASE_MODEL_IDN_WIDE_RESNET_101 = 'idn_wresnet101'
    BASE_MODEL_NISHI_RESNET_18 = 'nishi_resnet18'
    BASE_MODEL_ORIGINAL = 'original'


class IdentityClassifier(nn.Module):
    def __init__(self,
                 in_features: int = None,
                 out_features: int = None,
                 shared_part: bool = False,
                 layer_count: int = 3,
                 reverse_gradient: bool = None,
                 use_dgr: bool = True,
                 total_iterations: int = None):
        """
        The Identity Feature classifier for ASIF. There should be a dedicated IF head for each class in the dataset.

        The full IF head contains three layers. However, it is possible to split the IF head into a shared and private
        portion. This allows the shared portion to be trained on all data, while the private portion is only trained
        on a particular class' data. By default, the IF head will dedicate all three layers to the private portion.

        :param in_features: The number of dimensions of the input feature vector. If this is the private section of the
        IF head, this can be left out and the default vector size for that layer will be used.
        :param out_features: The number of dimensions of the output feature factor. If this is the public section of the
        IF head, this can be left out and the default vector size for that layer will be used.
        :param shared_part: If true, this is the public part of the IF head.
        :param layer_count: The number of layers to include. For the public section, this is the first X layers. For the
        private section, this is the last X layers.
        :param reverse_gradient: If true, reverse the gradient of the tensor on entry. If not provided, will
        automatically reverse the gradient for private IF heads.
        :param use_dgr: If true, use Dynamic Gradient Reversal. If not, use DANN.
        :param total_iterations: The total number of training iterations planned. Required if we are using DANN.
        """
        assert 3 >= layer_count >= 1, "Size must be in range [1, 3]"
        if shared_part or layer_count == 3:
            assert in_features is not None, "in_features must be specified for a public IF head."
        elif not shared_part or layer_count == 3:
            assert out_features is not None, "out_features must be specified for a private IF head."

        if not use_dgr:
            assert total_iterations is not None, "`total_iterations` must be provided if we are using DANN."

        if reverse_gradient is None:
            reverse_gradient = not shared_part

        super().__init__()

        self.shared_part = shared_part
        self.layer_count = layer_count
        self.reverse_gradient = reverse_gradient
        self.use_dgr = use_dgr
        self.total_iterations = total_iterations

        if shared_part:
            layers = []
            if layer_count >= 1:
                self.out_features = out_features if out_features is not None and layer_count == 1 else 1024
                layers += [nn.Linear(in_features=in_features, out_features=self.out_features)]
            if layer_count >= 2:
                self.out_features = out_features if out_features is not None and layer_count == 2 else 2048
                layers += [
                    nn.BatchNorm1d(num_features=1024),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(in_features=1024, out_features=self.out_features)]
            if layer_count >= 3:
                self.out_features = 1024
                layers += [
                    nn.BatchNorm1d(num_features=2048),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(in_features=2048, out_features=self.out_features)]
        else:
            layers = []
            self.out_features = out_features

            if layer_count >= 3:
                layers += [nn.Linear(in_features=in_features, out_features=1024)]
            if layer_count >= 2:
                layer_in_features = in_features if in_features is not None and layer_count == 2 else 1024
                layers += [
                    nn.BatchNorm1d(num_features=layer_in_features),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(in_features=layer_in_features, out_features=2048)]
            if layer_count >= 1:
                layer_in_features = in_features if in_features is not None and layer_count == 1 else 2048
                layers += [
                    nn.BatchNorm1d(num_features=layer_in_features),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(in_features=layer_in_features, out_features=out_features)]
        self.layer = nn.Sequential(*layers)

        self._downstream_heads = []  # type: List[IdentityClassifier]
        self._gradient_reversal_lambda = 1.
        if not shared_part:
            # We only reverse in the private section...
            self._dgr_goal = self._calc_goal_if_loss(out_features)
        else:
            self._dgr_goal = None

        self._initialize_weights()

    def clear_private_if_heads(self):
        """
        Removes any references to downstream IF heads. This should be called if resetting the IF heads.
        """
        self._downstream_heads.clear()

    def register_private_if_head(self, private_head: 'IdentityClassifier'):
        """
        If this is the shared public part of an IF head, you can register the private parts that take the output
        from this public part.

        By giving the public section knowledge of the private sections, the public section can calculate its gradient
        reversal lambda.

        :param private_head: A reference to a private IF head.
        """
        assert self.shared_part, 'This can only be called on a shared IF head.'
        assert not private_head.shared_part, '\'private_head\' must be a private IF head.'
        self._downstream_heads.append(private_head)

    def forward(self, identity_features: torch.Tensor) -> torch.Tensor:
        if self.reverse_gradient:
            identity_features = GradientReverse.apply(identity_features, self.calc_gradient_reversal_lambda())
        return self.layer(identity_features)

    def calc_gradient_reversal_lambda(self) -> float:
        """
        Returns the gradient reversal lambda value to use. In the case of a shared IF head, the lambda should be the
        average of the lambdas for all downstream private heads.
        :return: A lambda value.
        """
        if self.shared_part:
            assert len(self._downstream_heads) > 0, "The public head must have downstream private heads."
            total_lambda = 0.
            for _downstream_head in self._downstream_heads:
                total_lambda += _downstream_head.calc_gradient_reversal_lambda()
            return total_lambda / len(self._downstream_heads)
        else:
            return self._gradient_reversal_lambda

    def update_lambda(self, previous_average_loss: float, current_iteration: int):
        """
        Updates the lambda value used in the gradient reversal layer for a given identity feature classifier.
        There are two modes for lambda updates, depending on what is configured.
        In dynamic mode (DGR), the value is calculated using the average domain loss from the previous epoch. If this
        loss is too high, the domain classifier is allowed to improve. If it is too low, it is reversed.
        In static mode (DANN), the value is calculated based on the current progress through the training, irrespective
        of performance.
        :argument previous_average_loss The average group loss of the previous epoch.
        :argument current_iteration The current training iteration.
        """
        if self.use_dgr:
            # We hope to keep the group loss at this value, which corresponds to complete uncertainty.
            # We do this by setting the lambda on the gradient reversal layer. A positive 1.0 is the same as no gradient
            # reversal layer. -1 is a complete reversal.
            self._gradient_reversal_lambda = (previous_average_loss - self._dgr_goal) / self._dgr_goal
        else:
            self._gradient_reversal_lambda = \
                2 / (1 + math.exp(-10 * (current_iteration / (self.total_iterations + 1)))) - 1

    def _forward_unimplemented(self, *data: Any) -> None:
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _calc_goal_if_loss(count: int):
        """Calculates the loss that represents perfect uncertainty between 'count' choices."""
        criteria = nn.CrossEntropyLoss()
        test_data = torch.ones((1, count))
        target = torch.ones(1).type(torch.long)
        return criteria(test_data, target)


class AsifModel(nn.Module):
    def __init__(self,
                 data_dir: str,
                 in_channels: int,
                 class_count: int,
                 group_counts: List[int],
                 base_model_type: str,
                 use_log_softmax: bool,
                 shared_if_head_layer_count: int,
                 feature_vector_size: Optional[int],
                 reverse_shared_head_gradient: bool,
                 use_dgr: bool,
                 total_iterations: int):
        """
        Create a new ASIF model, based on the provided `base_model_type`.

        :param data_dir: The directory where pre-trained weights can be found.
        :param in_channels: The number of expected input channels.
        :param class_count: The number of classes to classify samples into.
        :param group_counts: The number of groups to classify group membership for. There will be one group classifier
        for each element in `group_counts` with the corresponding number of output possibilities.
        :param base_model_type: Determines the type of underlying feature extractor to use.
        :param use_log_softmax: If true, run the output through a log-softmax.
        :param shared_if_head_layer_count: The number of layers of the IF head that are shared across all classes.
        :param feature_vector_size: If provided, reduce the feature vector to this dimensionality.
        :param reverse_shared_head_gradient: If true, reverse the gradient on entering the shared IF head.
        :param use_dgr: If true, use Dynamic Gradient Reversal. If not, use DANN.
        :param total_iterations: The total number of training iterations planned. Required if we are using DANN.
        """
        super().__init__()

        self.in_channels = in_channels
        self.class_count = class_count
        self.group_counts = group_counts
        self.base_model_type = BaseModelType(base_model_type)
        self.use_log_softmax = use_log_softmax
        self.feature_vector_size = feature_vector_size if feature_vector_size is not None else 512
        self.narrow_feature_vector = (feature_vector_size is not None)
        self.shared_if_head_layer_count = shared_if_head_layer_count
        self.reverse_shared_head_gradient = reverse_shared_head_gradient
        self.data_dir = data_dir

        if self.base_model_type == BaseModelType.BASE_MODEL_RESNET_101:
            self.feature_extractor = self._create_resnet101()
        elif self.base_model_type == BaseModelType.BASE_MODEL_RESNET_18_PRETRAINED:
            self.feature_extractor = self._create_resnet18(True)
        elif self.base_model_type == BaseModelType.BASE_MODEL_RESNET_18_UNTRAINED:
            self.feature_extractor = self._create_resnet18(False)
        elif self.base_model_type == BaseModelType.BASE_MODEL_WIDE_RESNET_101:
            self.feature_extractor = WideResNet()
        elif self.base_model_type == BaseModelType.BASE_MODEL_IDN_WIDE_RESNET_101:
            self.feature_extractor = IdnWideResNet()
        elif self.base_model_type == BaseModelType.BASE_MODEL_NISHI_RESNET_18:
            self.feature_extractor = NishiResNet18(in_channels=self.in_channels)
        elif self.base_model_type == BaseModelType.BASE_MODEL_ORIGINAL:
            self.feature_extractor = OriginalFeatureExtractorModel(in_channels=self.in_channels)
        else:
            raise ValueError('{0} is not a supported option for feature extractor network.')

        if self.shared_if_head_layer_count > 0:
            self.shared_if_head = IdentityClassifier(in_features=self.feature_extractor.out_channels,
                                                     out_features=None,
                                                     shared_part=True,
                                                     reverse_gradient=self.reverse_shared_head_gradient,
                                                     layer_count=self.shared_if_head_layer_count,
                                                     use_dgr=use_dgr,
                                                     total_iterations=total_iterations)
            out_features = self.shared_if_head.out_features
        else:
            self.shared_if_head = None
            out_features = self.feature_extractor.out_channels

        if self.narrow_feature_vector:
            self.narrower = nn.Linear(in_features=out_features,
                                      out_features=self.feature_vector_size)
        else:
            self.feature_vector_size = out_features
        self.label_classifier = nn.Linear(in_features=self.feature_extractor.out_channels, out_features=class_count)

        self.group_classifier_count = len(group_counts)

        self._initialize_weights()

    def clear_private_if_heads(self):
        """
        Removes any references to downstream IF heads. This should be called if resetting the IF heads.
        """
        if self.shared_if_head is not None:
            self.shared_if_head.clear_private_if_heads()

    def register_private_if_head(self, private_head: IdentityClassifier):
        """
        Registers the private ASIF heads with the rest of the network.

        By giving the public section knowledge of the private sections, the public section can calculate its gradient
        reversal lambda.

        :param private_head: A reference to a private IF head.
        """
        if self.shared_if_head is not None:
            self.shared_if_head.register_private_if_head(private_head)

    def get_fc_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Gets the parameters for training the feature classifier."""
        if self.narrow_feature_vector:
            return chain(self.feature_extractor.parameters(),
                         self.narrower.parameters(),
                         self.label_classifier.parameters())
        else:
            return chain(self.feature_extractor.parameters(),
                         self.label_classifier.parameters())

    def get_shared_if_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Gets the shared parameters for training the IF heads."""
        if self.narrow_feature_vector and self.shared_if_head is not None:
            return chain(self.feature_extractor.parameters(),
                         self.narrower.parameters(),
                         self.shared_if_head.parameters())
        elif self.narrow_feature_vector and self.shared_if_head is None:
            return chain(self.feature_extractor.parameters(),
                         self.narrower.parameters())
        elif not self.narrow_feature_vector and self.shared_if_head is not None:
            return chain(self.feature_extractor.parameters(),
                         self.shared_if_head.parameters())
        elif not self.narrow_feature_vector and self.shared_if_head is None:
            return self.feature_extractor.parameters()
        else:
            # This should not be possible...
            raise RuntimeError('WTF?')

    def _create_resnet18(self, pretrained: bool) -> nn.Module:
        resnet_model = resnet18(pretrained=False)

        if pretrained:
            state_dict = torch.load(os.path.join(self.data_dir, 'resnet18-f37072fd.pth'))
            resnet_model.load_state_dict(state_dict)

        def _resnet_local_forward(x: torch.Tensor) -> torch.Tensor:
            """Override the default forward to remove the use of the FC"""
            x = resnet_model.conv1(x)
            x = resnet_model.bn1(x)
            x = resnet_model.relu(x)
            x = resnet_model.maxpool(x)

            x = resnet_model.layer1(x)
            x = resnet_model.layer2(x)
            x = resnet_model.layer3(x)
            x = resnet_model.layer4(x)

            x = resnet_model.avgpool(x)
            x = torch.flatten(x, 1)

            return x
        resnet_model.forward = _resnet_local_forward
        resnet_model.out_channels = resnet_model.fc.in_features
        return resnet_model

    @staticmethod
    def _create_resnet101() -> nn.Module:
        resnet_model = resnet101(pretrained=False)

        def _resnet_local_forward(x: torch.Tensor) -> torch.Tensor:
            """Override the default forward to remove the use of the FC"""
            x = resnet_model.conv1(x)
            x = resnet_model.bn1(x)
            x = resnet_model.relu(x)
            x = resnet_model.maxpool(x)

            x = resnet_model.layer1(x)
            x = resnet_model.layer2(x)
            x = resnet_model.layer3(x)
            x = resnet_model.layer4(x)

            x = resnet_model.avgpool(x)
            x = torch.flatten(x, 1)

            return x
        resnet_model.forward = _resnet_local_forward
        resnet_model.out_channels = resnet_model.fc.in_features
        return resnet_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs input through the model and returns the prediction label and feature vector.

        :param x: The input tensor.
        :return: A tuple containing:
         - label logits
         - Raw feature vector from the feature extractor
         - Feature vector to be fed into the IF heads
        """
        features = self.feature_extractor(x)
        features = functional.relu(features)
        fe_output_features = features.clone()
        labels = self.label_classifier(fe_output_features)

        if self.use_log_softmax:
            labels = functional.log_softmax(labels, dim=1)

        # Pass the features through the shared portion of the IF head, if there is any...
        if self.shared_if_head is not None:
            features = self.shared_if_head(features)

        if self.narrow_feature_vector:
            features = self.narrower(features)

        return labels, fe_output_features, features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
