#ifndef __INPUT_H__
#define __INPUT_H__

struct InputState {
    bool key_pressed_w = false;
    bool key_pressed_a = false;
    bool key_pressed_s = false;
    bool key_pressed_d = false;
    bool key_pressed_q = false;
    bool key_pressed_e = false;
    bool mouse_down = false;
    int prev_x, prev_y;
};

#endif  // __INPUT_H__