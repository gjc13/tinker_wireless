#!/usr/bin/env python

# Copyright (c) 2011, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Willow Garage, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import rospy
import sys

from geometry_msgs.msg import Pose2D, Twist
import tf
import geometry_msgs.msg

import sys, select, termios, tty
from detect_rssi import detect_rssi

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

V_X = 0.2
V_Y = 0.2
V_A = 0.2

RATE = 10.0

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('tinker_teleop')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)
    rate = rospy.Rate(RATE)
    listener = tf.TransformListener()

    try:
        target = Twist()
        target.linear.x = 0
        target.linear.y = 0
        target.linear.z = 0
        target.angular.x = 0
        target.angular.y = 0
        target.angular.z = 0

        while(1):
            key = getKey()
            if key == 'w':
                target.linear.x = V_X
            elif key == 's':
                target.linear.x = -V_X
            elif key == 'a':
                target.linear.y = V_Y
            elif key == 'd':
                target.linear.y = -V_Y
            elif key == 'z':
                target.angular.z = V_A
            elif key == 'c':
                target.angular.z = -V_A
            elif key == 'q':
                pub.publish(target)
                break
            else:
                target.linear.x = target.linear.y = target.angular.z = 0
            sys.stdout.write('\r')
            pub.publish(target)
            if key == ' ':
                try:
                    (trans,rot) = listener.lookupTransform('/base_link', '/map', rospy.Time(0))
                    print(trans)
                    for v in detect_rssi():
                        print(v)
                except:
                    pass

            #print("loop: {0}".format(count))
            #print("target: vx: {0}, wz: {1}".format(target_speed, target_turn))
            #print("publihsed: vx: {0}, wz: {1}".format(twist.linear.x, twist.angular.z))
            rate.sleep()
    except Exception as e:
        print e


    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

