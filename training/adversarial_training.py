
import configparser

from training.losses import loss_hinge_dis, compute_grad_reg

# Load config value
config = configparser.ConfigParser()
config.read('configs/training.cfg')
W2 = config.getfloat('Training', 'w2_adv_loss')


# _____________________________________________________________________________
def adversarial(discriminator, batch, img_l1, img_l2, img_l3, img_l4):
    '''
        Adversarial training for multi-scale adversarial loss
    '''
    # _____________________
    # Train with real data
    batch.color_sketch.requires_grad_() # necessary to calculate grad_reg
    dis_real = discriminator(batch.color_sketch)
    d_loss_real = 1/4 *(
        (loss_hinge_dis(discriminator(batch.l1_color_sketch), 'real') * 1) +
        (loss_hinge_dis(discriminator(batch.l2_color_sketch), 'real') * 2) +
        (loss_hinge_dis(discriminator(batch.l3_color_sketch), 'real') * 3) +
        (loss_hinge_dis(dis_real, 'real') * 10)
    )
    # Gradient Penalty Regularization for real images (color_sketch)
    grad_reg = W2 * compute_grad_reg(dis_real, batch.color_sketch).mean()

    # _____________________
    # Train with fake data
    d_loss_fake = 1/4 *(
        (loss_hinge_dis(discriminator(img_l1), 'fake') * 1) +
        (loss_hinge_dis(discriminator(img_l2), 'fake') * 2) +
        (loss_hinge_dis(discriminator(img_l3), 'fake') * 3) +
        (loss_hinge_dis(discriminator(img_l4), 'fake') * 10)
    )

    adv_loss = d_loss_real + d_loss_fake + grad_reg
    return adv_loss
