from setuptools import setup

package_name = 'spectrum_viz'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Simple realtime spectrum visualiser (ROS2)',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spectrum_gui = spectrum_viz.spectrum_gui:main',
            'spectrum_plot = spectrum_viz.spectrum_plot:main',
        ],
    },
)
