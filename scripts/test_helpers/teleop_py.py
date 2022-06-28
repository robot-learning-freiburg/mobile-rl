import rospy
from geometry_msgs.msg import Twist
import sys
from past.builtins import raw_input

twist = Twist()

def values():
    vel = float(input('velocity?'))
    # vel = 1.0
    print('(w for forward, a for left, s for reverse, d for right, k for turning left, l for turning right and . to exit)' + '\n')
    s = raw_input(':- ')
    if s[0] == 'w':
        twist.linear.x = vel
        twist.angular.z = 0.0
        twist.linear.y = 0.0
    elif s[0] == 's':
        twist.linear.x = -vel
        twist.angular.z = 0.0
        twist.linear.y = 0.0
    elif s[0] == 'd':
        twist.linear.y = -vel
        twist.angular.z = 0.0
        twist.linear.x = 0.0
    elif s[0] == 'a':
        twist.linear.y = vel
        twist.angular.z = 0.0
        twist.linear.x = 0.0
    elif s[0] == 'k':
        twist.angular.z = vel
        twist.linear.x = twist.linear.y = 0.0
    elif s[0] == 'l':
        twist.angular.z = -vel
        twist.linear.x = twist.linear.y = 0.0
    elif s[0] == '.':
        twist.angular.z = twist.linear.x = twist.linear.y = 0.0
        sys.exit()
    else:
        twist.linear.x = twist.linear.y = twist.angular.z = 0.0
        print('Wrong command entered \n')
    return twist

def keyboard():
    pub = rospy.Publisher('base_controller/command', Twist, queue_size=1)
    rospy.init_node('teleop_py', anonymous=True)
    rate = rospy.Rate(50)
    twist = values()

    while not rospy.is_shutdown():
        print(twist)
        pub.publish(twist)
        rate.sleep()

if __name__ == '__main__':
    try:
        keyboard()
    except rospy.ROSInterruptException:
        pass