   M S B A M L   `   `   `  ����D  @Pizzaria1, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null%clr-namespace:Pizzaria1	Pizzaria1  5      !    Pizzaria1.UserControlInicio   5      \ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5      R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5      Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5      @d2http://schemas.microsoft.com/expression/blend/2008     5      )localclr-namespace:Pizzaria1     5      U QMaterialDesignThemes.Wpf, Version=3.0.1.920, Culture=neutral, PublicKeyToken=nullFmaterialDesign1http://materialdesigninxaml.net/winfx/xaml/themes  5      $	��360��5	      $	��715��6     � 	Resources  5      ,  &Pizzaria1.ChatMessageListItemViewModel   chat(  "Pizzaria1.ChatMessageListViewModel  chatlist��%:   &	        &	        5   
   6-     5   
   6)   5      .��5      � .+�5   
   ?  9MaterialDesignThemes.Wpf.Transitions.TransitioningContent  -   5          Name$ TrainsitionigContentSlide��5   .   $��	0,0,0.2,0q�6�   $��2��6�   C  =MaterialDesignThemes.Wpf.Transitions.TransitioningContentBase   OpeningEffect 6Q   D  >MaterialDesignThemes.Wpf.Transitions.TransitionEffectExtension  *SlideInFromLeft+   Duration$ 0:0:0.8X�.��5      �  � ColumnDefinitions 5      �� 5      $	��530 �6'   �� 5      $��1* �6'    � RowDefinitions 5      �� 5      $��* �6$   �� 5      $
��Auto �6$   .+�5       � $��0��6    � DataContext# [ 5!      .+�5"      �� $
��Auto=�6#   .��5#      �� ��6'   �� ��5$      �� 5%   "   .R�5&   &   ��  J� RenderTransformOrigin$ 0.5,0.5&�6-   ._�5'   *   � .+�5)   .   �� $	��Top=�65   $
��Left=�5*   1   $��0,8,0,0q�5+   1    �� CornerRadius$ 20��5-   1   $��35��5.   1   $��35��5/   1   
�������$��1q�51   1   y�5,   1   �� *
SendByMe+	 �� 	Converter	 +   %Pizzaria1.BooleanToVisiblityConverter  ._�52   2   �� $��Resources/Olaf.jpg��68   $��5.2,4.2,6.4,3.4q�6T   54   /   � 56   .   $��
30,18,25,1q�63   ��57   /   �� 
 �� Path$
 SendByMe�	 ,  &Pizzaria1.SentByMeToAlignmentConverter   58   2   �� 59   6   $��* �6D   �� 5:   6   $
��Auto �6D   .+�5=   2   �� $ 10��69   
������� �� Padding$ 15q�5?   5   ._�5A   6   �� $��Center=�6@   $
��Wrap=�5B   <   ��5C   <   �� $
 Message�5D   3   F� 5E   2   $��1��67   
���   �$��
15 -1 15 0q�5G   3   +����        р�� ��� � -1    �        	$��0��5J   3   
���������5I   3   �� $
 SendByMe�	   5K   3   5M   /   5N   +   5O   '   5P   #   5R      5S      5T      �� 5Y      $��1��6    K� ZIndex$ 3��6&   $��50��67   $	��114��6F   # [  6V   ._�5Z      � .+�5\      �� -   5         ��Name$ MessageText��5\   "   $��0��6U   $��80��6e    �� TextWrapping$
 Wrap=�6t   $��10,3,95.6,7.6q�6�    �� AcceptsReturn .  �� 
AcceptsTab . " �� VerticalScrollBarVisibility$
 Auto=�6�   ��67   �� $
 Message�6�   �� 5`      -   5        $��8q�5`   "    �� 	IsDefault . |�.  $��Bottom=�6Q   $��446,0,9.6,7.6q�6�   .��5a      '  !MaterialDesignThemes.Wpf.PackIcon     Kind$
 Send=�66   5b      5c      5d      5g      5h      5i      5j      