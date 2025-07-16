import torch
import torch.nn as nn


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class SumFusion_DGL(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion_DGL, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):

        outx=self.fc_x(x)
        outy=self.fc_y(y)

        x_detach=x.detach()
        y_detach=y.detach()
        output = self.fc_x(x_detach) + self.fc_y(y_detach)
        return outx, outy, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)

        output = self.fc_out(output)
        return x, y, output


class ConcatFusion_DGL(nn.Module):
    def __init__(self, input_dim=512*2, output_dim=100):
        super(ConcatFusion_DGL, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.fc_auxi = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        
        output = torch.cat((x, y), dim=1)
        output=output.detach()

        output = self.fc_out(output)
        x_out = self.fc_out(torch.cat((x, torch.zeros_like(y)), dim=1))
        y_out = self.fc_out(torch.cat((torch.zeros_like(x), y), dim=1))
        return x_out, y_out, output


class ConcatFusion_DGL_unimodal(nn.Module):
    def __init__(self, input_dim=512*2, output_dim=100):
        super(ConcatFusion_DGL_unimodal, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.fc_auxi = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)

        output=output.detach()

        output = self.fc_out(output)
        x_out = self.fc_auxi(torch.cat((x, torch.zeros_like(y)), dim=1))
        y_out = self.fc_auxi(torch.cat((torch.zeros_like(x), y), dim=1))
        return x_out, y_out, output


class ConcatFusion_Swin(nn.Module):
    def __init__(self, input_dim=768 * 2, output_dim=100):
        super(ConcatFusion_Swin, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)

        output = self.fc_out(output)
        return x, y, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=768, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.fc = nn.Linear(dim * dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):
        # if self.x_film:
        #     film = x
        #     to_be_film = y
        # else:
        #     film = y
        #     to_be_film = x
        #
        # gamma, beta = torch.split(self.fc(film), self.dim, 1)
        #
        # output = gamma * to_be_film + beta
        x = torch.unsqueeze(x, dim=2)
        y = torch.unsqueeze(y, dim=1)
        z = torch.bmm(x, y)
        # print(z.shape)
        z = z.view(z.shape[0], -1)
        output = self.fc(z)
        output = self.fc_out(output)

        return x, y, output

class FiLM_DGL(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM_DGL, self).__init__()

        self.fc = nn.Linear(dim * dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):
        # if self.x_film:
        #     film = x
        #     to_be_film = y
        # else:
        #     film = y
        #     to_be_film = x
        #
        # gamma, beta = torch.split(self.fc(film), self.dim, 1)
        #
        # output = gamma * to_be_film + beta




        x = torch.unsqueeze(x, dim=2)
        y = torch.unsqueeze(y, dim=1)


        x_detach=x.detach()
        y_detach=y.detach()

        z = torch.bmm(x_detach, y_detach)
        # print(z.shape)
        z = z.view(z.shape[0], -1)
        output = self.fc(z)
        output = self.fc_out(output)

        z_x=torch.bmm(x, x.transpose(2,1))
        z_x = z_x.view(z_x.shape[0], -1)
        z_x = self.fc(z_x)
        z_x = self.fc_out(z_x)

        z_y=torch.bmm(y.transpose(2,1), y)
        z_y = z_y.view(z_y.shape[0], -1)
        z_y = self.fc(z_y)
        z_y = self.fc_out(z_y)

        return z_x, z_y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):

        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output


class GatedFusion_DGL(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion_DGL, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):

        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        x_detach=out_x.detach()
        y_detach=out_y.detach()


        if self.x_gate:
            gate = self.sigmoid(x_detach)
            output = self.fc_out(torch.mul(gate, y_detach))
        else:
            gate = self.sigmoid(y_detach)
            output = self.fc_out(torch.mul(x_detach, gate))

        gate_x=self.sigmoid(out_x)
        out_x=self.fc_out(torch.mul(gate_x,out_x))
        gate_y=self.sigmoid(out_y)
        out_y=self.fc_out(torch.mul(gate_y,out_y))
        return out_x, out_y, output
